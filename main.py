from demo import make_animation
import imageio
from skimage.transform import resize
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

pathToModel = "model.pth.tar"
pathToImage = "1.jpg"
pathToVideo = "1.mp4"


from demo import load_checkpoints
generator, kp_detector = load_checkpoints(config_path='config.yaml',
                                          checkpoint_path='pathToModel')

def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani

source_image = imageio.imread(pathToImage)
driving_video = imageio.mimread(pathToVideo, memtest=False)


#Resize image and video to 256x256

source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True,
                             adapt_movement_scale=True)

HTML(display(source_image, driving_video, predictions).to_html5_video())