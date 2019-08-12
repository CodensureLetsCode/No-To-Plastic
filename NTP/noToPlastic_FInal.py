import os
from flask import Flask,url_for,redirect,render_template, request,send_from_directory
from werkzeug import secure_filename
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
sys.path.append("..")
from PIL import Image
from tensorflow import DType
import math
import reverse_geocoder as rg 
import datetime
import MySQLdb
import time
from tensorflow import DType

from flask_mail import Mail, Message
import folium


UPLOAD_FOLDER='C:/Users/HP/Desktop/NTP/static/img/'
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'codensure.letscode@gmail.com'
app.config['MAIL_PASSWORD'] = 'Dimash2923'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True


mail=Mail(app)

MODEL_FILENAME = 'frozen_inference_graph.pb'
LABELS_FILENAME = 'labels.txt'

current_location=""
lat=''
lng=''
mapdict={}
mapbranddict={}
class ObjectDetection(object):
    """Class for Custom Vision's exported object detection model
    """

    ANCHORS = np.array([[0.573, 0.677], [1.87, 2.06], [3.34, 5.47], [7.88, 3.53], [9.77, 9.17]])
    IOU_THRESHOLD = 0.45

    def __init__(self, labels, prob_threshold = 0.10, max_detections = 20):
        """Initialize the class

        Args:
            labels ([str]): list of labels for the exported model.
            prob_threshold (float): threshold for class probability.
            max_detections (int): the max number of output results.
        """

        assert len(labels) >= 1, "At least 1 label is required"

        self.labels = labels
        self.prob_threshold = prob_threshold
        self.max_detections = max_detections

    def _logistic(self, x):
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def _non_maximum_suppression(self, boxes, class_probs, max_detections):
        """Remove overlapping bouding boxes
        """
        assert len(boxes) == len(class_probs)
        
        max_detections = min(max_detections, len(boxes))
        max_probs = np.amax(class_probs, axis=1)
        max_classes = np.argmax(class_probs, axis=1)

        areas = boxes[:,2] * boxes[:,3]

        selected_boxes = []
        selected_classes = []
        selected_probs = []

        while len(selected_boxes) < max_detections:
            # Select the prediction with the highest probability.
            i = np.argmax(max_probs)
            if max_probs[i] < self.prob_threshold:
                break

            # Save the selected prediction
            selected_boxes.append(boxes[i])
            selected_classes.append(max_classes[i])
            selected_probs.append(max_probs[i])

            box = boxes[i]
            other_indices = np.concatenate((np.arange(i), np.arange(i+1,len(boxes))))
            other_boxes = boxes[other_indices]

            # Get overlap between the 'box' and 'other_boxes'
            x1 = np.maximum(box[0], other_boxes[:,0])
            y1 = np.maximum(box[1], other_boxes[:,1])
            x2 = np.minimum(box[0]+box[2], other_boxes[:,0]+other_boxes[:,2])
            y2 = np.minimum(box[1]+box[3], other_boxes[:,1]+other_boxes[:,3])
            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)

            # Calculate Intersection Over Union (IOU)
            overlap_area = w * h
            iou = overlap_area / (areas[i] + areas[other_indices] - overlap_area)

            # Find the overlapping predictions
            overlapping_indices = other_indices[np.where(iou > self.IOU_THRESHOLD)[0]]
            overlapping_indices = np.append(overlapping_indices, i)

            # Set the probability of overlapping predictions to zero, and udpate max_probs and max_classes.
            class_probs[overlapping_indices,max_classes[i]] = 0
            max_probs[overlapping_indices] = np.amax(class_probs[overlapping_indices], axis=1)
            max_classes[overlapping_indices] = np.argmax(class_probs[overlapping_indices], axis=1)

        assert len(selected_boxes) == len(selected_classes) and len(selected_boxes) == len(selected_probs)
        return selected_boxes, selected_classes, selected_probs

    def _extract_bb(self, prediction_output, anchors):
        assert len(prediction_output.shape) == 3
        num_anchor = anchors.shape[0]
        height, width, channels = prediction_output.shape
        assert channels % num_anchor == 0

        num_class = int(channels / num_anchor) - 5
        assert num_class == len(self.labels)

        outputs = prediction_output.reshape((height, width, num_anchor, -1))
        
        # Extract bouding box information
        x = (self._logistic(outputs[...,0]) + np.arange(width)[np.newaxis, :, np.newaxis]) / width
        y = (self._logistic(outputs[...,1]) + np.arange(height)[:, np.newaxis, np.newaxis]) / height
        w = np.exp(outputs[...,2]) * anchors[:,0][np.newaxis, np.newaxis, :] / width
        h = np.exp(outputs[...,3]) * anchors[:,1][np.newaxis, np.newaxis, :] / height

        # (x,y) in the network outputs is the center of the bounding box. Convert them to top-left.
        x = x - w / 2
        y = y - h / 2
        boxes = np.stack((x,y,w,h), axis=-1).reshape(-1, 4)

        # Get confidence for the bounding boxes.
        objectness = self._logistic(outputs[...,4])

        # Get class probabilities for the bounding boxes.
        class_probs = outputs[...,5:]
        class_probs = np.exp(class_probs - np.amax(class_probs, axis=3)[..., np.newaxis])
        class_probs = class_probs / np.sum(class_probs, axis=3)[..., np.newaxis] * objectness[..., np.newaxis]
        class_probs = class_probs.reshape(-1, num_class)

        assert len(boxes) == len(class_probs)
        return (boxes, class_probs)
    
    def predict_image(self, image):
        inputs = self.preprocess(image)
        prediction_outputs = self.predict(inputs)
        return self.postprocess(prediction_outputs)

    def preprocess(self, image):
        image = image.convert("RGB") if image.mode != "RGB" else image
        image = image.resize((416, 416))
        return image

    def predict(self, preprocessed_inputs):
        """Evaluate the model and get the output

        Need to be implemented for each platforms. i.e. TensorFlow, CoreML, etc.
        """
        raise NotImplementedError

    def postprocess(self, prediction_outputs):
        """ Extract bounding boxes from the model outputs.

        Args:
            prediction_outputs: Output from the object detection model. (H x W x C)

        Returns:
            List of Prediction objects.
        """
        boxes, class_probs = self._extract_bb(prediction_outputs, self.ANCHORS)

        # Remove bounding boxes whose confidence is lower than the threshold.
        max_probs = np.amax(class_probs, axis=1)
        index, = np.where(max_probs > self.prob_threshold)
        index = index[(-max_probs[index]).argsort()]

        # Remove overlapping bounding boxes
        selected_boxes, selected_classes, selected_probs = self._non_maximum_suppression(boxes[index], class_probs[index], self.max_detections)

        return [{'probability': round(float(selected_probs[i]), 8),
                 'tagId': int(selected_classes[i]),
                 'tagName': self.labels[selected_classes[i]],
                 'boundingBox': {
                     'left': round(float(selected_boxes[i][0]), 8),
                     'top': round(float(selected_boxes[i][1]), 8),
                     'width': round(float(selected_boxes[i][2]), 8),
                     'height': round(float(selected_boxes[i][3]), 8)
                 }
             } for i in range(len(selected_boxes))]


class Brand_detection(ObjectDetection):

	def __init__(self, graph_def, labels):
		super(Brand_detection, self).__init__(labels)
		self.graph = tf.Graph()
		with self.graph.as_default():
			tf.import_graph_def(graph_def, name='')
			
	def predict(self, preprocessed_image):
		inputs = np.array(preprocessed_image, dtype=np.float)[:,:,(2,1,0)] # RGB -> BGR

		with tf.Session(graph=self.graph) as sess:
			output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
			outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis,...]})
			return outputs[0]


def reverseGeocode(coordinates): 
    result = rg.search(coordinates) 
      
    # result is a list containing ordered dictionary. 
    return result

def connection():
	conn = MySQLdb.connect(host="localhost",
		user="aditi",
		passwd="ads296",
		db="ntp")

	c=conn.cursor()
	return c, conn


@app.route("/")
def login():
    return render_template('userform.html')

@app.route('/home',methods=['POST'])
def savepost():
	usr = request.form['user']
	return render_template('home.html',username=usr)


@app.route('/home/results',methods=['POST'])
def uploadImage():
	pic=request.files['pic']
	
	j=0
	
	category_index={1: {'id': 1, 'name': 'Balaji'}, 2: {'id': 2, 'name': 'CocaCola'}, 3: {'id': 3, 'name': 'Kurkure'}, 4: {'id': 4, 'name': 'Lays'}, 5: {'id': 5, 'name': 'Sprite'}}
	classes=[0,0,0,0,0]
	classes1=[]
	scores=[0.,0.,0.,0.,0.]
	scores1=[]
	if request.method == 'POST':
		f = request.files['pic']
		filename = secure_filename(f.filename)
		f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

		graph_def_obj = tf.GraphDef()
		with tf.gfile.FastGFile(MODEL_FILENAME, 'rb') as f:
			graph_def_obj.ParseFromString(f.read())

		# Load labels
		with open(LABELS_FILENAME, 'r') as f:
			labels = [l.strip() for l in f.readlines()]

		od_model = Brand_detection(graph_def_obj, labels)
		image = Image.open(UPLOAD_FOLDER+filename)
		imageWhole = image
		imagenp=np.array(image,dtype=np.uint8)
		predictions1 = od_model.predict_image(image)
		print(predictions1)
		print(type(predictions1))
		pred_out_list=[]
		pred_prob=0.0
		draw = ImageDraw.Draw(imageWhole)
		
		for p in predictions1:
			if(p['probability']>0.35):
				pred_prob=str(round(p['probability']*100,2))
				pred_out_list.append((p['tagName'],pred_prob))
		print(pred_out_list)

		orgInfo=image.info

		startX=0.0
		startY=0.0
		endX=0.0
		endY=0.0
		widthP,heightP=image.size
		c=0
		d=0
		w=0
		h=0
		boxes=[]
		boxes1=[]
		print("WIDTH:",widthP,"\nHEIGHT:",heightP)
		for d1 in predictions1:
			if(d1['probability']>0.35):
				print(d1['boundingBox'].items())
				for n1,n2 in d1['boundingBox'].items():
					#image=imageWhole
					if(n1=='left'):
						startX=float(n2)*widthP
						l=float(n2)
					elif(n1=='top'):
						startY=float(n2)*heightP
						t=float(n2)
					elif(n1=='width'):
						endX=startX+float(n2)*widthP
						w=float(n2)
					elif(n1=='height'):
						endY=startY+float(n2)*heightP
						h=float(n2)
			print("left: ",startX,"\ntop: ",startY,"\nwidth: ",endX,"\nheight: ",endY)
			#boxes.append([startX,startY,endX,endY])
			draw.rectangle(((startX, startY), (endX, endY)), fill=None, width=10, outline="red")
	
		try:
			c, conn = connection()
			print("okay")
		except Exception as e:
			print(str(e))
		
		current_datetime=datetime.datetime.now()
		print(current_datetime)
		formatted_date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
		print("fd: ",formatted_date)

		time.sleep(10)
		for tup in pred_out_list:
			try:
				global current_location
				#sql="CREATE TABLE IF NOT EXISTS brands_track (dateAndtime datetime, brand varchar(25), location varchar(100))"
				loc=current_location #returnLoc()
				time.sleep(5)
				print("hey!"+loc)
				sql_insert="INSERT INTO brands_track(dateAndtime,brand,location) VALUES('%s','%s','%s')"%(formatted_date,tup[0],loc)
				c.execute(sql_insert)
				conn.commit()

			except Exception as e:
				print(str(e))

		print(bool(orgInfo))
		if bool(orgInfo)!=False:
			imageWhole.save("C:/Users/HP/Desktop/NTP/static/img/output_{}".format(filename),"JPEG")

		else:
			imageWhole.save("C:/Users/HP/Desktop/NTP/static/img/output_{}".format(filename),"JPEG")
		msg1='Predictions'
		
		drawGraph()
		
		time.sleep(5)
	return render_template('result.html',pic='output_{}'.format(filename),msg1=msg1,brands=pred_out_list)

@app.route("/home/results/mail_acknowledgement", methods=['POST'])
def send_mail():
	email_id=request.form['user_email']
	msg = Message('No To Plastic Result', sender = 'codensure.letscode@gmail.com', recipients = [email_id])
	msg.body = "Hello! This is the latest record of Plastic Waste Brands in different locations."
	with app.open_resource("C:/Users/HP/Desktop/NTP/static/img/brand_graph2.png") as fp:
		msg.attach("brand_graph2.png", "image/png", fp.read())
	mail.send(msg)
	return render_template('mail_ack.html')

@app.route('/home/results/Thankyou')
def Thankyou():
	print ('I got clicked!')
	return render_template('Thankyou.html')

@app.route('/home/results/Complaint')
def Complaint():
	print ('I got clicked!')
	return render_template('Complaint.html')

@app.route('/home/results/Complaint/LabelImg')
def label():
	print ('I got clicked!')
	
	return render_template('Thankyou.html')

@app.route('/home/results/map')
def showmap():
	global lat, lng
	r= lat	
	r2= lng	
	print(r,r2)
	loclatlng={'Thane':[19.2183, 72.9781],'Nerul':[19.0338, 73.0196],
				'Panvel':[19.004735, 73.1221789],'Kalundri':[18.9798, 73.1254],
				'Mumbai':[19.0760, 72.8777],'Airoli':[19.1590, 72.9986]}
	# Make an empty map
	m = folium.Map(location=[r, r2], tiles="OpenStreetMap", zoom_start=10)
	for i,j in mapdict.items():
		print(i,j)
		if(i in loclatlng.keys()):
			
			iconDustbin=folium.features.CustomIcon('./static/img/dustbin.png',icon_size=(30,30)) 
			folium.Marker(location=[loclatlng[i][0],loclatlng[i][1]], popup="<strong>{0}</strong><br>{1}_{2}<br>{3}_{4}<br>{5}_{6}<br>{7}_{8}<br>{9}_{10}".format(i,j[0][0],j[0][1],j[1][0],j[1][1],j[2][0],j[2][1],j[3][0],j[3][1],j[4][0],j[4][1]), tooltip="Click to see info about plastic waste", icon=iconDustbin).add_to(m)
			
	# Save it as html
	m.save('./templates/no_to_plastic_map.html')
	time.sleep(10)
	return render_template('no_to_plastic_map.html')



@app.route('/home/results/location',methods=['POST'])
def storeLocation():
	global lat, lng
	loc=""
	dict_loc={}
	global current_location
	if request.method == 'POST':
		user_location = request.get_json()
		print(user_location)
		lat=user_location['Latitude']
		lng=user_location['Longitude']
		coordinates=(lat,lng)
		loct=reverseGeocode(coordinates)
		print("loct: ",loct)
		for l in loct:
			print(l)
			dict_loc=dict(l)
			loc=dict_loc['name']
			current_location=loc
			print(current_location)
	return ''

def returnLoc():
	global current_location
	return current_location


def drawGraph():
	global mapdict,mapbranddict

	distinct_locations=[]
	distinct_brands=[]
	brands_records={}
	brands_records_percent={}
	loc_brands={}
	loc_brands_percent={}
	means_brnd_dict={}
	mapitems=[]
	try:
		c, conn = connection()
		sql_dist="SELECT DISTINCT location FROM brands_track"
		c.execute(sql_dist)
		distinct_locations=list(c.fetchall())
		print(distinct_locations)
		sql_brands="SELECT DISTINCT brand FROM brands_track"
		c.execute(sql_brands)
		distinct_brands=list(c.fetchall())
		print(distinct_brands)
		for t1 in distinct_locations:
			total_packs=0
			for b1 in distinct_brands:
				brand_count=0
				sql_retrieve="SELECT COUNT(brand) FROM brands_track WHERE brand='%s' AND location='%s'"%(str(b1[0]),str(t1[0]))
				c.execute(sql_retrieve)
				brand_count=c.fetchone()
				#print(brand_count[0])
				brands_records[b1[0]]=brand_count[0]
				total_packs=total_packs+brand_count[0]
				mapbranddict[b1[0]]=total_packs
			mapitems=list(mapbranddict.items())
			mapdict[t1[0]]=mapitems
			print(brands_records)
			loc_brands[t1[0]]=brands_records
			print("total_packs ",total_packs)
			for br,cnt in brands_records.items():
				brands_records_percent[br]=(cnt/total_packs)*100
			print("brands_records_percent ",brands_records_percent)
			d=0
			for brnd,pr in brands_records_percent.items():
				if(brnd==distinct_brands[d][0]):
					if(brnd in means_brnd_dict):
						means_brnd_dict[brnd]=means_brnd_dict[brnd]+(pr,)
						d=d+1
					else:
						means_brnd_dict[brnd]=(pr,)
						d=d+1
					
			print("means_brnd_dict: ",means_brnd_dict)
			loc_brands_percent[t1[0]]=brands_records_percent
			print(loc_brands)
			print("loc_brands_percent: ",loc_brands_percent)

		
		print("MAP: ",mapdict)
		# data to plot
		n_groups = len(loc_brands_percent.items())
		print(n_groups)
		fig, ax = plt.subplots()
		color_list=['blue','red','green','black','cyan']
		gap = .8 / len(means_brnd_dict)
		i=0
		for yb,row in means_brnd_dict.items():
			X = np.arange(len(row))
			rects=plt.bar(X + i * gap, row, width = gap, color = color_list[i % len(color_list)], label=yb)
			i=i+1

		loc_list=[]
		for l in distinct_locations:
			loc_list.append(l[0])

		print(loc_list)

		plt.xlabel('Location')
		plt.ylabel('Brands Percentage')
		plt.title('Percentage of Brands in Different Locations')
		plt.xticks(np.arange(n_groups) + 0.3, tuple(loc_list))
		plt.legend()


		brndPic="C:/Users/HP/Desktop/NTP/static/img/brand_graph2.png"
		if os.path.exists(brndPic):
			os.remove(brndPic)
			fig.savefig("C:/Users/HP/Desktop/NTP/static/img/brand_graph2.png")
		else:
			fig.savefig("C:/Users/HP/Desktop/NTP/static/img/brand_graph2.png")

		print("drawGraph executed")
	except Exception as e:
		print(str(e))
	

if __name__ == '__main__':
	
	app.run(debug=True)