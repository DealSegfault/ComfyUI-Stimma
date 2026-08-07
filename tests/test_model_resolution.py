"""Tests for resolving bundled workflow model names against ComfyUI paths."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stp_server.discovery import _resolve_model_combo_value, _validate_workflow


class TestModelResolution(unittest.TestCase):
    def test_exact_match_is_preserved(self):
        resolved, ambiguous = _resolve_model_combo_value(
            "Anima/model.safetensors", ["Anima/model.safetensors"]
        )
        self.assertEqual(resolved, "Anima/model.safetensors")
        self.assertEqual(ambiguous, [])

    def test_unique_nested_basename_is_resolved(self):
        resolved, ambiguous = _resolve_model_combo_value(
            "model.safetensors",
            ["Flux/other.safetensors", "Anima/model.safetensors"],
        )
        self.assertEqual(resolved, "Anima/model.safetensors")
        self.assertEqual(ambiguous, [])

    def test_windows_separator_is_supported(self):
        resolved, ambiguous = _resolve_model_combo_value(
            "model.safetensors", [r"Anima\model.safetensors"]
        )
        self.assertEqual(resolved, r"Anima\model.safetensors")
        self.assertEqual(ambiguous, [])

    def test_duplicate_basename_is_ambiguous(self):
        resolved, ambiguous = _resolve_model_combo_value(
            "model.safetensors",
            ["Anima/model.safetensors", "Archive/model.safetensors"],
        )
        self.assertIsNone(resolved)
        self.assertEqual(len(ambiguous), 2)

    def test_validation_rewrites_prompt_to_comfyui_path(self):
        prompt = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "anima-preview3-base.safetensors"},
            }
        }
        object_info = {
            "UNETLoader": {
                "input": {
                    "required": {
                        "unet_name": ([r"Anima\anima-preview3-base.safetensors"],)
                    }
                }
            }
        }

        warnings = _validate_workflow(prompt, object_info)

        self.assertEqual(warnings, [])
        self.assertEqual(
            prompt["1"]["inputs"]["unet_name"],
            r"Anima\anima-preview3-base.safetensors",
        )

    def test_anima_models_resolve_from_separate_nested_external_roots(self):
        prompt = {
            "clip": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": "qwen_3_06b_base.safetensors"},
            },
            "vae": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "qwen_image_vae.safetensors"},
            },
            "unet": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "anima-preview3-base.safetensors"},
            },
        }
        object_info = {
            "CLIPLoader": {"input": {"required": {
                "clip_name": ([r"Anima\qwen_3_06b_base.safetensors"],),
            }}},
            "VAELoader": {"input": {"required": {
                "vae_name": ([r"Anima\qwen_image_vae.safetensors"],),
            }}},
            "UNETLoader": {"input": {"required": {
                "unet_name": ([r"Anima\anima-preview3-base.safetensors"],),
            }}},
        }

        warnings = _validate_workflow(prompt, object_info)

        self.assertEqual(warnings, [])
        self.assertEqual(
            prompt["clip"]["inputs"]["clip_name"],
            r"Anima\qwen_3_06b_base.safetensors",
        )
        self.assertEqual(
            prompt["vae"]["inputs"]["vae_name"],
            r"Anima\qwen_image_vae.safetensors",
        )
        self.assertEqual(
            prompt["unet"]["inputs"]["unet_name"],
            r"Anima\anima-preview3-base.safetensors",
        )

    def test_validation_blocks_ambiguous_basename(self):
        prompt = {
            "1": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "vae.safetensors"},
            }
        }
        object_info = {
            "VAELoader": {
                "input": {
                    "required": {
                        "vae_name": ([
                            "Anima/vae.safetensors",
                            "Qwen/vae.safetensors",
                        ],)
                    }
                }
            }
        }

        warnings = _validate_workflow(prompt, object_info)

        self.assertEqual(len(warnings), 1)
        self.assertIn("ambiguous matches", warnings[0])
        self.assertEqual(prompt["1"]["inputs"]["vae_name"], "vae.safetensors")


if __name__ == "__main__":
    unittest.main()
