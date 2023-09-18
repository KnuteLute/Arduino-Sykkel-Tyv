from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os




def check_face(embedder,detector,recognizer,le,func_confidence):
	"""
	embedder = model_embedding
	recognizer = model_classification
	face detection based from KnutKnut = detector
	remember that this function takes in lables called le.

	"""
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

	fps = FPS.start()

	while True:
		# grab the frame from the threaded video stream
		frame = vs.read()
		# resize the frame to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		frame = imutils.resize(frame, width=600)
		(h, w) = frame.shape[:2]
		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)
		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()
		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]
			# filter out weak detections
			if confidence > func_confidence:
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				# extract the face ROI
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]
				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue
				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()
				# perform classification to recognize the face
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]
				# draw the bounding box of the face along with the
				# associated probability
				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
							  (0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		# update the FPS counter
		fps.update()
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the `a` key was pressed, break from the loop
		if key == ord("a"):
			break
	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()



check_face()



