"""Focused tests for descriptor construction from Stimma workflow fields.

Run: python tests/test_tool_builder.py
"""

import json
import os
import sys
import types
import unittest
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

package = types.ModuleType("stp_server")
package.__path__ = [os.path.join(ROOT, "stp_server")]
sys.modules["stp_server"] = package
config_module = types.ModuleType("stp_server.config")
config_module.Config = type("Config", (), {})
sys.modules["stp_server.config"] = config_module

from stp_server.discovery import DiscoveredWorkflow
from stp_server.tool_builder import _build_single_tool


def field(node_id, class_type, inputs):
    return {"node_id": node_id, "class_type": class_type, "inputs": inputs}


class TestReferenceToVideoDescriptor(unittest.TestCase):
    def test_typed_sections_are_optional_but_one_is_required(self):
        fields = [field("prompt", "StimmaPromptParam", {
            "name": "prompt", "default_text": "", "required": True, "ui_order": 0,
        })]
        fields.extend(
            field(f"image-{i}", "StimmaImageParam", {
                "required": False, "ui_control": "image_picker", "ui_order": i,
            })
            for i in range(1, 3)
        )
        fields.extend(
            field(f"video-{i}", "StimmaVideoParam", {
                "required": False, "ui_control": "video_picker", "ui_order": 10 + i,
            })
            for i in range(1, 3)
        )
        fields.extend(
            field(f"audio-{i}", "StimmaAudioParam", {
                "required": False, "ui_control": "audio_picker", "ui_order": 20 + i,
                "ui_label": "Standalone Reference Audio", "audio_role": "reference",
            })
            for i in range(1, 3)
        )
        workflow = DiscoveredWorkflow(
            file_path="reference.json",
            api_prompt={},
            tool_info={
                "slug": "h3-reference",
                "display_name": "H3 Reference",
                "task_types": ["reference-to-video"],
                "description": "",
            },
            field_nodes=fields,
        )

        descriptor = _build_single_tool(
            workflow, object_info=None, config=object(), provider=object()
        ).to_descriptor()
        schema = descriptor.parameter_schema

        self.assertEqual(schema["anyOf"], [
            {"required": ["input_images"], "properties": {"input_images": {"minItems": 1}}},
            {"required": ["input_videos"], "properties": {"input_videos": {"minItems": 1}}},
            {"required": ["input_audios"], "properties": {"input_audios": {"minItems": 1}}},
        ])
        self.assertNotIn("input_images", schema["required"])
        self.assertEqual(schema["properties"]["input_images"]["x-min-items"], 0)
        self.assertEqual(schema["properties"]["input_images"]["x-max-items"], 2)
        self.assertEqual(schema["properties"]["input_videos"]["x-min-items"], 0)
        self.assertEqual(schema["properties"]["input_videos"]["x-max-items"], 2)
        self.assertEqual(schema["properties"]["input_audios"]["x-min-items"], 0)
        self.assertEqual(schema["properties"]["input_audios"]["x-max-items"], 2)
        self.assertIn("immediately before", schema["properties"]["input_videos"]["description"])
        self.assertIn("Numbered after", schema["properties"]["input_audios"]["description"])


class TestReferenceToVideoWorkflow(unittest.TestCase):
    def test_links_and_h3_socket_order_match_native_model_presentation(self):
        workflow_path = Path(ROOT) / "workflows" / "Stimma-MiniMax-H3-R2V.json"
        workflow = json.loads(workflow_path.read_text())
        nodes = {node["id"]: node for node in workflow["nodes"]}

        for link_id, source, source_slot, target, target_slot, _type in workflow["links"]:
            self.assertIn(link_id, nodes[source]["outputs"][source_slot]["links"])
            self.assertEqual(nodes[target]["inputs"][target_slot]["link"], link_id)

        h3 = next(
            node for node in workflow["nodes"]
            if node["type"] == "MiniMaxH3ReferenceToVideo"
        )
        expected = ["clip", "vae", "audio_vae", "prompt", "width", "height", "length", "ref_image_size"]
        expected.extend(f"ref_images.ref_image_{index}" for index in range(9))
        for index in range(3):
            expected.extend((
                f"ref_videos.ref_video_{index}",
                f"ref_video_audios.ref_video_audio_{index}",
            ))
        expected.extend(f"ref_audios.ref_audio_{index}" for index in range(3))

        self.assertEqual([item["name"] for item in h3["inputs"]], expected)

        field_nodes = [
            node for node in workflow["nodes"]
            if node["type"] in {"StimmaImageParam", "StimmaVideoParam", "StimmaAudioParam"}
        ]
        self.assertEqual(sum(node["type"] == "StimmaImageParam" for node in field_nodes), 9)
        self.assertEqual(sum(node["type"] == "StimmaVideoParam" for node in field_nodes), 3)
        self.assertEqual(sum(node["type"] == "StimmaAudioParam" for node in field_nodes), 3)
        for node in field_nodes:
            # Slot 1 is the new required widget for typed visual fields and was
            # already the required widget for audio fields.
            self.assertIs(node["widgets_values"][1], False)
        for node in field_nodes:
            if node["type"] == "StimmaVideoParam":
                self.assertEqual(node["inputs"][-1]["name"], "target_fps")
                self.assertEqual(node["widgets_values"][-1], 24)


if __name__ == "__main__":
    unittest.main()
