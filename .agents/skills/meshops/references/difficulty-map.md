# Difficulty → guard map

Every implementing track should cite lesson numbers in DoD.

| Lesson | Failure | MeshOps requirement |
|--------|---------|---------------------|
| §1 | Left/right / character ambiguity | Fixed cameras; label figures; ask if low confidence |
| §2 | Thickness ≠ roundness (sheet limb) | sheet_score; T3 route; front silhouette acceptance |
| §3 | High poly / perf | Indexed working mesh; proxy LOD; local ROI ops |
| §4 | Blender CDN 403 | Mirror list + cached portable; track 0010 |
| §5 | Boolean / MC origin-space | Volume checks; world transform; bbox regression |
| §6 | Boolean wipeout 358KB | Export guards face/size; hard fail |
| §7 | Over-delete sheet → torso holes | No auto linked-flat delete; prefer reshape |
| §8 | Global remesh melts detail | Forbid whole-model voxel remesh on hero |
| §9 | Weak viz / VLM lies | F3D ortho+depth first-class |
| §10 | Sheet→tube incomplete | Preview only; Blender escalate |
| §11 | FreeCAD wrong tool for organic | FreeCAD = T5 only |
| §12 | False numeric success | Visual QA mandatory |
| §13 | Expectation vs automation | Honest escalation messaging |

Fixture policy: `fixtures/rogue2/` + `fixtures/synthetic/` (sheet, two-figure, clothing false-positive).
