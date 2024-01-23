import cv2
import numpy as np
import pdb

from PIL import Image, ImageDraw, ImageFont

class Cv2TextRenderer:
  '''
    image - must be a cv2 Image in BGR format
  '''
  def render(self, string, image, top=0, left=0, font_face=cv2.FONT_HERSHEY_COMPLEX, pixel_height=10, color=(0,0,0), thickness=1):
    cv2.putText(image, 
                string,
                (left, top),
                font_face,
                cv2.getFontScaleFromHeight(font_face, pixel_height, thickness),
                color,
                thickness)

class PILTextRenderer:
  DEFAULT_FONT_FACE="SourceSans3-Regular.ttf"
  DEFAULT_FONT_PATH="assets/fonts/"
  
  def __init__(self, fonts=DEFAULT_FONT_FACE, font_path=DEFAULT_FONT_PATH):
    self.fonts = {}
    self.font_path = font_path
    if isinstance(fonts, str):
      fonts = [fonts]
      
    for font in fonts:
      self.lazy_load_font(font, font_path)
  
  def lazy_load_font(self, font_face, font_path=None):
    this_font = self.fonts.get(font_face)
    if this_font is None:
      self.fonts[font_face] = ImageFont.truetype( (font_path or self.font_path) + font_face )
      this_font = self.fonts[font_face]
  
    return this_font

  def pixel_width(self, string, image, pixel_height, font_face=DEFAULT_FONT_FACE, features=None):
    '''
      Return the width in pixels of the given string, when rendered in the given font 
      on the given image, at the given pixel height with the given features
    '''
    # hack - textlength() only works with an actual ImageDraw object, so just return
    return len(string) * pixel_height
    
  def render(self, string, image, top=0, left=0, font=None, font_face=DEFAULT_FONT_FACE, align='left', pixel_height=10, color=(0,0,0), thickness=1):
    '''
      image - must be a PIL image in RGB format
    '''
    draw = ImageDraw.Draw(image)
    font_to_use = font or self.lazy_load_font(font_face)
    pdb.set_trace()
    draw.text((top, left),
              string,
              fill=color,
              font=font_to_use,
              align=align,
              font_size=pixel_height)
    pdb.set_trace()
    return image

  def convert_and_render(self, string, image, top=0, left=0, font=None, font_face=DEFAULT_FONT_FACE, color='#FFF'):
    # def draw_text_with_pillow(image, text, origin, color='#FFF'):
    # convert color format to PIL-compatible
    cv2_im_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
     # Pass the image to PIL  
    pil_im = Image.fromarray(cv2_im_rgb)
    # draw the text
    draw = ImageDraw.Draw(pil_im)
    draw.text((top, left), string, font=(font or self.lazy_load_font(font_face)), fill=color)
    # convert it back to OpenCV format
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)