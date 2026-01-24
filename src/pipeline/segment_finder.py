import os
from typing import List


def find_seg_dirs(subject_dir: str) -> List[str]:
    """
    Recursively find all segment directories (seg*, test*)
    under a subject fMRI directory.
    """
    seg_dirs = []

    for root, dirs, _ in os.walk(subject_dir):
        for d in dirs:
            if d.startswith(("seg", "test")):
                seg_dirs.append(os.path.join(root, d))

    return sorted(seg_dirs)
