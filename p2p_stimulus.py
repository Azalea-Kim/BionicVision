#haven't tried yet
import numpy as np
import pulse2percept as p2p
import cv2

input_video = "videos/baseline_combination_clip_quad.mp4"
output_folder = "p2p-combo-5/"
grayscale_video = "videos/baseline_combination_clip_quad_gray.mp4"



# First, let's check the video properties
cap = cv2.VideoCapture(input_video)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Original video: {width}x{height}, {total_frames} frames at {fps} fps")

# Define new dimensions (reduce to 25% of original size)
new_width = width // 4
new_height = height // 4

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(grayscale_video, fourcc, fps, (new_width, new_height), isColor=False)

# Process frames
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (new_width, new_height))
    out.write(resized)
    
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count}/{total_frames} frames")

cap.release()
out.release()



for RHO in [100, 300, 500]:
    for LAM in [0, 100, 200]:

        if LAM == 0:
            model = p2p.models.ScoreboardModel(xrange=(-10, 10), yrange=(-10, 10), rho=RHO)
        else:
            model = p2p.models.AxonMapModel(xrange=(-10, 10), yrange=(-10, 10), rho=RHO, axlambda=LAM)
        model.build()

        grid_sizes = [(8, 8), (16, 16), (32, 32)]
        implants = {}
        for gsize in grid_sizes:
            # Fit all electrodes into (-2000, 2000):
            spacing = 4000 / gsize[0]
            # Sensible radius might be 1/5th of spacing:
            radius = spacing / 5
            egrid = p2p.implants.ElectrodeGrid(gsize, spacing,
                                               etype=p2p.implants.DiskElectrode,
                                               r=radius)
            implants['%dx%d' % gsize] = p2p.implants.ProsthesisSystem(egrid)

        current_video = p2p.stimuli.VideoStimulus(grayscale_video, as_gray=True)
        for gsize in grid_sizes:
            res = gsize[0]
            implant_key = str(res) + "x" + str(res)
            implant = implants[implant_key]
            implant.stim = current_video.resize(implant.earray.shape)
            percept = model.predict_percept(implant)
            percept.save(output_folder + "sample_{}({},{})".format(res, RHO, LAM) + ".mp4", fps=20)  # You can control the frame rate with fps=
