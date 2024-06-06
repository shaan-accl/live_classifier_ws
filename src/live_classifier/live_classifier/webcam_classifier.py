import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from vision_msgs.msg import Classification, ObjectHypothesis

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from timeit import default_timer as timer

class WebcamClassifier(Node):

    def __init__(self):
        super().__init__('webcam_classification')
        # Create a subscriber to the Image topic
        self.image_subscriber = self.create_subscription(Image, 'image', self.listener_callback, 10)
        self.image_subscriber

        # create a publisher onto the vision_msgs 2D classification topic
        self.classification_publisher = self.create_publisher(Classification, 'classification', 10)

        # create a model parameter, by default the model is resnet18
        # self.declare_parameter('model', "resnet18")
        # model_name = self.get_parameter('model')
        
        # print(model_name.value)

        # Use the SqueezeNet model for classification
        self.classification_model = self.create_classification_model()

        # Load it onto the GPU and set it to evaluation mode
        self.classification_model.eval().cuda()

        # Use CV bridge to convert ROS Image to CV_image for visualizing in window
        self.bridge = CvBridge()

        # Find the location of the ImageNet labels text and open it
        # with open(os.getenv("HOME") + '/ros2_models/imagenet_classes.txt') as f:
        #    self.labels = [line.strip() for line in f.readlines()]
        
        self.labels = ['CRACK', 'DENT_NICK', 'EROSION', 'RUB', 'TBC LOSS', 'TEAR']
 
    

    def create_classification_model(self):
        
        # if(str(model_name.value) == "squeezenet"):
        #     return torchvision.models.squeezenet1_1(pretrained=True)
        
        # elif(str(model_name.value) == "alexnet"):
        #     return torchvision.models.alexnet(pretrained=True)
            
        # elif(str(model_name.value) == "resnet18"):
        #     return torchvision.models.resnet18(pretrained=True)
            
        # elif(str(model_name.value) == "resnet50"):
        #     return torchvision.models.resnet50(pretrained=True)
            
        # print("Invalid model selection. Select amongst alexnet, squeezenet, resnet18 and resnet50")
        return torch.jit.load("/home/accl_orin_nano2/OCAST/traced_model.pt")
      


 
    def classify_image(self,img):
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomRotation(10),
            # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        tensor_to_image = transforms.ToPILImage()
        img = tensor_to_image(img)
        img_t = transform(img).cuda()
        batch_t = torch.unsqueeze(img_t, 0)
	
        # Classify the image
        start = timer() 
        out = self.classification_model(batch_t)
        end = timer()

        print("Live classifier time: ", (end-start))

        _, index = torch.max(out, 1)

        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        return self.labels[index[0]] , percentage[index[0]].item()
        

    def listener_callback(self, msg):
        
        img_data = np.asarray(msg.data)
        img = np.reshape(img_data,(msg.height, msg.width, 3))
        start = timer()
        classified, confidence = self.classify_image(img)
        end = timer()
        time = str(end-start)
        to_display = "Classification: " + classified + " ,confidence: " + str(confidence) + " time: " + time
        self.get_logger().info(to_display) 

        # Definition of Classification2D message
        classification = Classification()
        classification.header = msg.header
        result = ObjectHypothesis()
        result.class_id = classified
        result.score = confidence
        classification.results.append(result)

        # Publish Classification results
        self.classification_publisher.publish(classification)
      
        # Use OpenCV to visualize the images being classified from webcam 
        try:
          cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
          print(e)
        # cv2.imshow('webcam_window', cv_image)
        # Custom window
        cv2.namedWindow('Webcam Window', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Webcam Window', cv_image)
        cv2.resizeWindow('Webcam Window', 700, 500)
        cv2.waitKey(1)
        
def main(args=None):
    rclpy.init(args=args)

    webcam_classifier = WebcamClassifier()

    rclpy.spin(webcam_classifier)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    webcam_classifier.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()