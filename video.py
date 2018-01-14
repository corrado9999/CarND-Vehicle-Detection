import model
from moviepy.editor import VideoFileClip
import click

@click.command()
@click.argument('input-file', type=click.Path())
@click.argument('output-file')
@click.option('--subclip', '-s', type=(float, float), default=(None, None))
@click.option('--codec', '-c', default=None)
def main(input_file, output_file, subclip=None, codec=None):
    clip = VideoFileClip(input_file)
    if subclip != (None, None):
        clip = clip.subclip(*subclip)
    output = clip.fl_image(model.tracker)
    output.write_videofile(output_file, audio=False, codec=codec)

if __name__ == '__main__':
    main()
