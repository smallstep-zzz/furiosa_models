import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
from torch import fx
from compiler_engine.graph_lowerings import lower_scaled_dot_product_attention, manual_scaled_dot_product_attention, dtype_promoting_cat
from compiler_engine.device_fallback import replace_graph_device
from compiler_engine.renegade_diffusion_compiler import _CPUComponent, _NPUBlock, _segment_unet_for_npu, _segment_clip_text_encoder, keep_scheduler_latents_on_cpu

class UnetSegmentationTests(unittest.TestCase):
    def test_tensor_metadata_records_structure_without_values(self):
        from compiler_engine.renegade_diffusion_compiler import _tensor_metadata
        summary = _tensor_metadata(torch.ones(2, 3, dtype=torch.bfloat16))
        self.assertEqual(summary["shape"], [2, 3])
        self.assertEqual(summary["dtype"], "torch.bfloat16")
        self.assertNotIn("values", summary)

    def test_top_level_unet_blocks_are_individually_wrapped(self):
        class Block(torch.nn.Module):
            def forward(self, value):
                return value
        class Unet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_in = Block()
                self.down_blocks = torch.nn.ModuleList([Block(), Block()])
                self.mid_block = Block()
                self.up_blocks = torch.nn.ModuleList([Block()])
        unet = Unet()
        with patch("torch.compile", side_effect=lambda module, backend: module):
            count = _segment_unet_for_npu(unet, "unit")
        self.assertEqual(count, 5)
        self.assertIsInstance(unet.conv_in, _NPUBlock)
        self.assertIsInstance(unet.down_blocks[0], _NPUBlock)
        self.assertIsInstance(unet.mid_block, _NPUBlock)
        self.assertIsInstance(unet.up_blocks[0], _NPUBlock)

    def test_group_norm_mean_square_decomposition_matches_torch(self):
        from compiler_engine.graph_lowerings import GroupNormMeanSquare
        generator = torch.Generator().manual_seed(81)
        source = torch.nn.GroupNorm(32, 320, dtype=torch.bfloat16)
        value = torch.randn(1, 320, 8, 8, dtype=torch.bfloat16, generator=generator)
        expected = F.group_norm(value, 32, source.weight, source.bias, source.eps)
        actual = GroupNormMeanSquare(source)(value)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)



class SchedulerLatentPlacementTests(unittest.TestCase):
    def test_scheduler_latents_stay_on_cpu(self):
        class Pipeline:
            def prepare_latents(self, *args, **kwargs):
                return args, kwargs
        pipe = Pipeline()
        keep_scheduler_latents_on_cpu(pipe)
        args, kwargs = pipe.prepare_latents(1, 4, 512, 512, torch.bfloat16, torch.device("furiosa:0"))
        self.assertEqual(args[5], torch.device("cpu"))
        self.assertEqual(kwargs, {})
        args, kwargs = pipe.prepare_latents(device=torch.device("furiosa:0"))
        self.assertEqual(kwargs["device"], torch.device("cpu"))


class CpuComponentTests(unittest.TestCase):
    def test_explicit_cpu_component_normalizes_inputs(self):
        class CpuOnly(torch.nn.Module):
            def forward(self, value):
                if value.device.type != "cpu":
                    raise RuntimeError("input was not moved to CPU")
                return value + 1
        value = torch.ones(3)
        result = _CPUComponent(CpuOnly())(value)
        self.assertEqual(result.device.type, "cpu")
        torch.testing.assert_close(result, torch.full((3,), 2.0))


class Conv2dDecompositionTests(unittest.TestCase):
    def test_conv2d_unfold_decomposition_matches_torch(self):
        from compiler_engine.graph_lowerings import conv2d_via_unfold
        x = torch.randn(1, 4, 16, 16)
        weight = torch.randn(8, 4, 3, 3)
        bias = torch.randn(8)
        expected = F.conv2d(x, weight, bias, stride=1, padding=1)
        actual = conv2d_via_unfold(x, weight, bias, stride=1, padding=1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_sdxl_spatial_matmul_decomposition_matches_torch(self):
        from compiler_engine.graph_lowerings import conv2d_via_spatial_matmul
        generator = torch.Generator().manual_seed(47)
        x = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16, generator=generator)
        weight = torch.randn(320, 4, 3, 3, dtype=torch.bfloat16, generator=generator)
        bias = torch.randn(320, dtype=torch.bfloat16, generator=generator)
        expected = F.conv2d(x, weight, bias, stride=1, padding=1)
        actual = conv2d_via_spatial_matmul(x, weight, bias, stride=1, padding=1)
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)

    def test_conv2d_decomposition_rejects_dilation_and_groups(self):
        from compiler_engine.graph_lowerings import conv2d_via_unfold
        x = torch.randn(1, 4, 8, 8)
        weight = torch.randn(4, 1, 3, 3)
        with self.assertRaisesRegex(NotImplementedError, "groups=1"):
            conv2d_via_unfold(x, weight, groups=4)


class SdpaLoweringTests(unittest.TestCase):
    def inputs(self, query_heads=4, kv_heads=4, q_len=5, kv_len=5):
        generator = torch.Generator().manual_seed(7)
        return (torch.randn(2, query_heads, q_len, 8, generator=generator), torch.randn(2, kv_heads, kv_len, 8, generator=generator), torch.randn(2, kv_heads, kv_len, 8, generator=generator))

    def assert_matches(self, query, key, value, **kwargs):
        expected = F.scaled_dot_product_attention(query, key, value, **kwargs)
        actual = manual_scaled_dot_product_attention(query, key, value, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_mixed_dtype_cat_matches_torch_promotion(self):
        values = [torch.ones(2, dtype=torch.bfloat16), torch.full((2,), 2.0, dtype=torch.float32)]
        expected = torch.cat(values)
        actual = dtype_promoting_cat(values)
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(actual, expected)


    def test_plain(self): self.assert_matches(*self.inputs())
    def test_boolean_mask(self):
        q,k,v=self.inputs(q_len=4,kv_len=6)
        self.assert_matches(q,k,v,attn_mask=torch.tensor([[1,1,0,0,1,0]],dtype=torch.bool).expand(4,6))
    def test_additive_mask(self):
        q,k,v=self.inputs(q_len=4,kv_len=6)
        self.assert_matches(q,k,v,attn_mask=torch.randn(4,6)*0.1)
    def test_causal(self): self.assert_matches(*self.inputs(),is_causal=True)
    def test_gqa(self): self.assert_matches(*self.inputs(query_heads=4,kv_heads=2),enable_gqa=True)
    def test_rejects_causal_mask_combination(self):
        q,k,v=self.inputs()
        with self.assertRaisesRegex(ValueError,'cannot combine'):
            manual_scaled_dot_product_attention(q,k,v,attn_mask=torch.ones(5,5,dtype=torch.bool),is_causal=True)
    def test_fx_pass_replaces_fused_sdpa(self):
        class Attention(torch.nn.Module):
            def forward(self,q,k,v): return F.scaled_dot_product_attention(q,k,v)
        gm=fx.symbolic_trace(Attention())
        self.assertEqual(lower_scaled_dot_product_attention(gm),1)
        self.assertTrue(any(n.op=='call_function' and n.target is manual_scaled_dot_product_attention for n in gm.graph.nodes))
        q,k,v=self.inputs()
        torch.testing.assert_close(gm(q,k,v),F.scaled_dot_product_attention(q,k,v),rtol=1e-5,atol=1e-6)
    def test_cpu_fallback_rewrites_explicit_furiosa_device(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        index = graph.call_function(torch.arange, args=(4,), kwargs={"device": "furiosa:0"})
        result = graph.call_function(torch.add, args=(x, index))
        graph.output(result)
        gm = fx.GraphModule(torch.nn.Module(), graph)
        replace_graph_device(gm, "cpu")
        x = torch.ones(4)
        torch.testing.assert_close(gm(x), torch.arange(4, dtype=x.dtype) + 1)


if __name__ == '__main__': unittest.main()


class StrictNpuModeTests(unittest.TestCase):
    def test_npu_only_mode_raises_on_strict_compile_failure(self):
        import os
        from compiler_engine.renegade_diffusion_compiler import _audited_backend
        with patch.dict(os.environ, {"FURIOSA_REQUIRE_NPU": "1"}):
            with patch("compiler_engine.renegade_diffusion_compiler._STRICT_BACKEND", side_effect=RuntimeError("no lowering")):
                with patch("compiler_engine.renegade_diffusion_compiler._NPU_EAGER_BACKEND") as eager:
                    compiled = torch.compile(torch.nn.ReLU(), backend=_audited_backend("unit", "strict_test"))
                    with self.assertRaisesRegex(RuntimeError, "strict compilation failed"):
                        compiled(torch.ones(1))
                    eager.assert_not_called()


class ClipSegmentationTests(unittest.TestCase):
    def test_public_clip_submodules_are_segmented(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        class TextModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = torch.nn.Identity()
                self.encoder = Encoder()
                self.final_layer_norm = torch.nn.Identity()
        class Clip(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.text_model = TextModel()
                self.text_projection = torch.nn.Identity()
        clip = Clip()
        with patch("torch.compile", side_effect=lambda module, backend: module):
            count = _segment_clip_text_encoder(clip, "unit", "text_encoder")
        self.assertEqual(count, 5)
        self.assertIsInstance(clip.text_model.embeddings, _NPUBlock)
        self.assertIsInstance(clip.text_model.encoder.layers[1], _NPUBlock)
        self.assertIsInstance(clip.text_model.final_layer_norm, _NPUBlock)
        self.assertIsInstance(clip.text_projection, _NPUBlock)
