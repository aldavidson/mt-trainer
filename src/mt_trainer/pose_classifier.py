import os
import pathlib

from json.decoder import JSONDecodeError

from mt_trainer.quantified_pose import QuantifiedPose


class PoseClassifier:
  
    def __init__(self, pose_archetypes=None, data_dir=None):
        self.pose_archetypes = pose_archetypes or {}
        if data_dir:
            self.load_training_data(data_dir)

    def load_training_data(self, dir_path):
        '''
            Load training data from the given dir_path.
            Training data must be in the form of a
            JSON-serialised QuantifiedPose, organised
            into a sub-folder for each technique. 
        '''
        for technique in QuantifiedPose.TECHNIQUES:
            archetype = QuantifiedPose({}, {}, {})

            technique_dir = os.path.join(dir_path, technique)
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
        '''
            TODO: This method doesn't belong in this class
        '''
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

    def classify(self, pose, threshold=0.9, max_results=1):
        '''
            Returns a list of most-similar poses to the given
            QuantifiedPose, and their cosine-similarity. 
            The list is sorted in descending order of similarity,
            and will contain at most max_results members.
            All members must have cosine similarity equal to or 
            greater than the given threshold.
        '''
        poses = self.similarities(pose)
        poses_over_threshold = list(
          (k, v) for k, v in poses.items() if v and v >= threshold
        )
        return sorted(poses_over_threshold, key=lambda v: v[1], reverse=True)[0:max_results]
