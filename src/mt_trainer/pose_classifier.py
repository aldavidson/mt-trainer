import os
import pathlib

from json.decoder import JSONDecodeError

from mt_trainer.quantified_pose import QuantifiedPose


class PoseClassifier:
  
    def __init__(self, pose_archetypes=None, data_dir=None):
        self.pose_archetypes = pose_archetypes or {}
        if data_dir:
            self.load_training_data(data_dir)

    def load_training_data(self, dir):
        for technique in QuantifiedPose.TECHNIQUES:
            archetype = QuantifiedPose({}, {}, {})

            technique_dir = os.path.join(dir, technique)
            files = PoseClassifier.files_in(technique_dir)

            files_loaded = 0
            for file in files:
                try:
                    pose = QuantifiedPose.load(file)
                    archetype = archetype.plus(pose)
                    files_loaded += 1
                except JSONDecodeError:
                    pass

            if files_loaded > 1:
                archetype.multiply_by(1.0 / files_loaded)

            self.pose_archetypes[technique] = archetype

    @staticmethod
    def files_in(dir):
        return [
            os.path.join(dirpath, f)
            for (dirpath, dirnames, filenames) in os.walk(dir)
            for f in filenames
        ]

    def similarities(self, pose):
        similarities = {}
        for technique, archetype in self.pose_archetypes.items():
            similarities[technique] = pose.similarity_to(archetype)

        return similarities

    def classify(self, pose, threshold=0.9):
        poses = self.similarities(pose)
        poses_over_threshold = list(
          (k, v) for k, v in poses.items() if v and v >= threshold
        )
        return sorted(poses_over_threshold, key=lambda v: v[1])
