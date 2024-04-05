class Layout:

    def __init__(self,
                 video_size,
                 annotation_panel_size,
                 world_landmarks_panel_size=None,
                 padding=2
                 ):
      self.video_size = video_size
      self.annotation_panel_size = annotation_panel_size
      self.world_landmarks_panel_size = world_landmarks_panel_size
      
      self.total_width = video_size[0] + padding + annotation_panel_size[0]
      
      # default layout:
      # -------------------------------------
      # |video           | annotation panel |
      # -------------------------------------
      #
      # But if we're given world_landmarks, we'll lay it out like this:
      # -------------------------------------
      # |video           | annotation panel |
      # |world_landmarks | (empty space)    |
      # -------------------------------------
      if self.world_landmarks_panel_size:
          self.total_height = video_size[1] + padding + world_landmarks_panel_size[1]
      else:
          self.total_height = video_size[1]

      pass

