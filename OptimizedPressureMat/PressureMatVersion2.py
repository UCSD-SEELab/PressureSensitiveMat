#Author: Aron Laszik
#License: SeeLab?
#Using: Adafruit_Python_ADS1x15 public domain github

MQTT_EN = False
TRAINING_EN = True
PLOTTING_EN = False
PRINT_EN = True
NP_EN = True

import time
#Import event handlers
import sys
import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#Import ADC's Module
sys.path.insert(0,"/home/pi/OptimizedPressureMat/libs/Adafruit_Python_ADS1x15/Adafruit_ADS1x15/")
print(sys.path)
import ADS1x15 as Adafruit_ADS1x15
print
#print ADS1x15
print
#Import RPIO GPIO Module
import RPi.GPIO as GPIO
#import csv to create the files with data
import csv
#import numpy to save as a numpy file
import numpy as np


if (MQTT_EN):
    import paho.mqtt.client as mqtt
    import json

#Set up the directory In which all the persons footprint data will enter
elif (TRAINING_EN):
    try:
        file_name = raw_input("What do you want the data file name to be?")
    except:
        print("Something Went Wrong")
        sys.exit()


#Import Plotting module import matplot.pyplot as pyplot import numpy
#Set up the boardd numbering system
GPIO.setmode(GPIO.BCM)


#Set up the board's output ports 19,26, and 16 to operate binary for first Decoder
#Set up the board's output ports 5,6, and 13 to operate binary for second Decoder
#27 inhibits the first decoder, while 17 inhibits the second decoder
channel_list = [13,19,26,6,5,27,17]
inhibless = [26,19,13]
GPIO.setup(channel_list,GPIO.OUT)


#Get rid of magic numbers, label the output ports
#by their orientation in the decoder 27,17,5,6
C7 = [27]
C6 = [26,27]
C5 = [19,27]
C4 = [19,26,27]
C3 = [13,27]
C2 = [13,26,27]
C1 = [13,19,27]
C0 = [13,19,26,27]
C15 = [17]
C14 = [26,17]
C13 = [19,17]
C12 = [19,26,17]
C11 = [13,17]
C10 = [13,26,17]
C9 = [13,19,17]
C8 = [13,19,26,17]
C23 = [5]
C22 = [26,5]
C21 = [19,5]
C20 = [19,26,5]
C19 = [13,5]
C18 = [13,26,5]
C17 = [13,19,5]
C16 = [13,19,26,5]
C31 = [6]
C30 = [26,6]
C29 = [19,6]
C28 = [19,26,6]
C27 = [13,6]
C26 = [13,26,6]
C25 = [13,19,6]
C24 = [13,19,26,6]
COL_LIST = [C0,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26,C27,C28,C29,C30,C31]
INHIB_A = 27
INHIB_B = 17
INHIB_C = 5
INHIB_D = 6


#GND,SDA,SCL,VDD
#Make a 16 bit ADC instance
#the address is specific to the high voltage being on and
#the busnum on all RPI's is 1
adc1 = Adafruit_ADS1x15.ADS1115(address=0x48,busnum=1)
adc2 = Adafruit_ADS1x15.ADS1115(address=0x4A,busnum=1)
adc3 = Adafruit_ADS1x15.ADS1115(address=0x4B,busnum=1)
adc4 = Adafruit_ADS1x15.ADS1115(address=0x49,busnum=1)


#setup the adc read specifics
#set gain to 1 to utilize full 3.3V input
#set data_rate to 860 to utilize full read speeds
GAIN = 1
DATA_RATE = 860
DIVIDER = 1400
adc1.opt_config_setup_0(GAIN,DATA_RATE)
adc1.opt_config_setup_1(GAIN,DATA_RATE)
adc1.opt_config_setup_2(GAIN,DATA_RATE)
adc1.opt_config_setup_3(GAIN,DATA_RATE)
adc2.opt_config_setup_0(GAIN,DATA_RATE)
adc2.opt_config_setup_1(GAIN,DATA_RATE)
adc2.opt_config_setup_2(GAIN,DATA_RATE)
adc2.opt_config_setup_3(GAIN,DATA_RATE)
adc3.opt_config_setup_0(GAIN,DATA_RATE)
adc3.opt_config_setup_1(GAIN,DATA_RATE)
adc3.opt_config_setup_2(GAIN,DATA_RATE)
adc3.opt_config_setup_3(GAIN,DATA_RATE)
adc4.opt_config_setup_0(GAIN,DATA_RATE)
adc4.opt_config_setup_1(GAIN,DATA_RATE)
adc4.opt_config_setup_2(GAIN,DATA_RATE)
adc4.opt_config_setup_3(GAIN,DATA_RATE)

#Initialize the numpy array which shall store our data
NUM_ROWS = 16
NUM_COLS = 32
COLLECT_LENGTH = 1140
data = np.zeros((NUM_ROWS,NUM_COLS,COLLECT_LENGTH))
timestamps = np.zeros(COLLECT_LENGTH)

#Set up the figure, 
if (PLOTTING_EN):
    pass
    #figure = pyplot.figure()
    #map = figure.add_subplot(111)
    #im = map.imshow(numpy.random.random(2,4))
    #plot.show(block=False)

#Make a function to read a columns readings
#TODO Switch to numpy Array
def readColumn(column, data_array, time_index,col_num):
    #Setup the GPIO output
    GPIO.output([INHIB_A,INHIB_B,INHIB_C,INHIB_D],GPIO.HIGH)
    GPIO.output(inhibless,GPIO.LOW)
    #time.sleep(0.005) #Probably not necessary
    GPIO.output(column,GPIO.HIGH)
    #time.sleep(0.005) #Probably not necessary
    if INHIB_A in column:
        GPIO.output(INHIB_A, GPIO.LOW)
    elif INHIB_B in column:
        GPIO.output(INHIB_B,GPIO.LOW)
    elif INHIB_C in column:
        GPIO.output(INHIB_C,GPIO.LOW)
    else:
        GPIO.output(INHIB_D,GPIO.LOW)
    time.sleep(0.0001)
    #Send out requests to read the voltages to each value
    adc1.send_read_request_0()
    adc2.send_read_request_0()
    adc3.send_read_request_0()
    adc4.send_read_request_0()
    #sleep until adc has had time to make read
    time.sleep(1.0/DIVIDER)
    #retrieve values
    data_array[0,col_num,time_index] = adc1.retrieve_read()
    data_array[4,col_num,time_index] = adc2.retrieve_read()
    data_array[8,col_num,time_index] = adc3.retrieve_read()
    data_array[12,col_num,time_index] = adc4.retrieve_read()
    #send out next bacth of requests
    adc1.send_read_request_1()
    adc2.send_read_request_1()
    adc3.send_read_request_1()
    adc4.send_read_request_1()
    #sleep until adc has had time to make read
    time.sleep(1.0/DIVIDER)
    #retrieve values
    data_array[1,col_num,time_index] = adc1.retrieve_read()
    data_array[5,col_num,time_index] = adc2.retrieve_read()
    data_array[9,col_num,time_index] = adc3.retrieve_read()
    data_array[13,col_num,time_index] = adc4.retrieve_read()
    #send out 3rd bacth of read requests
    adc1.send_read_request_2()
    adc2.send_read_request_2()
    adc3.send_read_request_2()
    adc4.send_read_request_2()
    #sleep until adc has had time to make read
    time.sleep(1.0/DIVIDER)
    #retrieve values
    data_array[2,col_num,time_index] = adc1.retrieve_read()
    data_array[6,col_num,time_index] = adc2.retrieve_read()
    data_array[10,col_num,time_index] = adc3.retrieve_read()
    data_array[14,col_num,time_index] = adc4.retrieve_read()
    #send out last batch of read requests
    adc1.send_read_request_3()
    adc2.send_read_request_3()
    adc3.send_read_request_3()
    adc4.send_read_request_3()
    #sleep until adc has had time to make read
    time.sleep(1.0/DIVIDER)
    #retrieve values
    data_array[3,col_num,time_index] = adc1.retrieve_read()
    data_array[7,col_num,time_index] = adc2.retrieve_read()
    data_array[11,col_num,time_index] = adc3.retrieve_read()
    data_array[15,col_num,time_index] = adc4.retrieve_read()
    return


#if __name__ == '__main__':
##    Boot up delay
#    t0 = time.time()
#    
#    while (time.time() - t0 < 15):
#        time.sleep(1)
        
#    # Initialize MQTT
#    if (MQTT_EN):
#        broker_address = 'server.healthyaging'
#        broker_port = 61613

#        # Setup MQTT connection
#        client = mqtt.Client(client_id='PressureMat', 
#                             clean_session=True,
#                             protocol=mqtt.MQTTv31) #additional parameters for clean_session, userdata, protection,
#        client.username_pw_set('admin', 'IBMProject$')
#        print('MQTT: Connecting to broker {0}:{1}'.format(broker_address, broker_port))
#        client.connect(host=broker_address, port=broker_port)
#        client.loop_start() #start the loop
#        topic_data = 'PressureMat/raw'
#    else:
#        client = None
#    t_last_mqtt_msg = time.time()
#    t_mqtt_msg_period = 1"""


print('ost mqtt setup')
#Run the sensor
#define the time counter
try:
    for frame in range(COLLECT_LENGTH):
        #Read columns and get timing data
        before = time.time()
        for i in range(32):
            #may be slow
            readColumn(COL_LIST[i],data,frame,i)
            timestamps[frame]=time.time()
        after = time.time()

        #write the data into a csv file
        #filename = data_dir + '/' + str(after) + '.csv'
        #with open(filename,'w') as csvFile:
        #    writer = csv.writer(csvFile)
        #    writer.writerows(values)
        #csvFile.close()


        #Print read values
        for i in range(25): 
            print


        #ASCII DATA!!!
        dict_results = {'timestamp':time.time()}
        for i in range(16):
            result = ''
            result_disp = ''
            for j in range(32):
                if  data[i,j,frame] < 400:
                    char = " "
                elif data[i,j,frame] < 800:
                    char = "."
                elif data[i,j,frame] < 1000:
                    char = "*"
                elif data[i,j,frame] < 1300:
                    char = "O"
                else:
                    char = "@"
                    #char = str(j[i])
                result += char + ' '
                result_disp += char + char + '  '
            key_name = 'result{0:02d}'.format(i)
            dict_results[key_name] = j
            #TODO verbose_EN
            print('Row' + '{0:2d}'.format(i) + ' ' + result_disp)
            print('      ' + result_disp)
        print
        print
        print(after - before)

        #Dump JSON Data
        if(MQTT_EN):
            if (not(client is None) & (time.time() - t_last_mqtt_msg > t_mqtt_msg_period)):
                try:
                    msg = json.dumps(dict_results)
                    client.publish(topic_data, msg)
                    t_last_mqtt_msg = time.time()
                except Exception as e:
                    print(e)
    #save out numpy array as binary data
    np.save(file_name + '.npy', data)
    np.save(file_name + 'Timestamps.npy', data)
    
    #Handle a ^C exit
except KeyboardInterrupt:
    adc1.stop_adc()
    adc2.stop_adc()
    adc3.stop_adc()
    adc4.stop_adc()
    GPIO.cleanup()

    #save out numpy array as binary data
    np.save(file_name + '.npy', data)
    np.save(file_name + 'Timestamps.npy', data)
        
    if (MQTT_EN):
        client.loop_stop()
    
    sys.exit(0)
