import model
from moviepy.editor import VideoFileClip
import click

@click.command()
@click.argument('input-file', type=click.Path())
@click.argument('output-file')
@click.option('--subclip', '-s', type=(float, float), default=(None, None))
@click.option('--codec', '-c', default=None)
@click.option('--heat-thresholds', '-H', type=(int,int), default=(0,0))
@click.option('--fir-length', '-L', type=int, default=0)
def main(input_file, output_file, subclip=None, codec=None, heat_thresholds=(0,0), fir_length=0):
    clip = VideoFileClip(input_file)
    if subclip != (None, None):
        clip = clip.subclip(*subclip)
    for i,h in enumerate(heat_thresholds):
        if h:
            model.tracker.heat_thresholds[i] = h
    if fir_length:
        model.tracker.fir_length = fir_length
    output = clip.fl_image(model.tracker)
    output.write_videofile(output_file, audio=False, codec=codec)

if __name__ == '__main__':
    main()
