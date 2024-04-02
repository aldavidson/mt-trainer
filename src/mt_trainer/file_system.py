import os
import pathlib

class FileSystem:
    '''
      Convenience wrappers around File System operations
    '''
    @staticmethod
    def files_in(directory):
        '''
            TODO: This method doesn't belong in this class
        '''
        return [
            os.path.join(dirpath, f)
            for (dirpath, dirnames, filenames) in os.walk(directory)
            for f in filenames
        ]
