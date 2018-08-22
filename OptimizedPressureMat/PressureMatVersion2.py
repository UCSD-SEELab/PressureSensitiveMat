#Author: Aron Laszik
#License: SeeLab?
#Using: Adafruit_Python_ADS1x15 public domain github

MQTT_EN = True

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


if (MQTT_EN):
    import paho.mqtt.client as mqtt
    import json
    
#Import Plotting module import matplot.pyplot as pyplot import numpy
#Set up the boardd numbering system
GPIO.setmode(GPIO.BCM)

print("CHECK1")

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
INHIB_A = 27
INHIB_B = 17
INHIB_C = 5
INHIB_D = 6
print("CHECK2")
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
print("CHECK3")
#Set up the figure, 
#figure = pyplot.figure()
#map = figure.add_subplot(111)
#im = map.imshow(numpy.random.random(2,4))
#plot.show(block=False)

#Make a function to read a columns readings
def readColumn(column):
    #Setup the GPIO output
    GPIO.output([INHIB_A,INHIB_B,INHIB_C,INHIB_D],GPIO.HIGH)
    GPIO.output(inhibless,GPIO.LOW)
    time.sleep(0.005)
    GPIO.output(column,GPIO.HIGH)
    time.sleep(0.005)
    if INHIB_A in column:
        GPIO.output(INHIB_A, GPIO.LOW)
    elif INHIB_B in column:
        GPIO.output(INHIB_B,GPIO.LOW)
    elif INHIB_C in column:
        GPIO.output(INHIB_C,GPIO.LOW)
    else:
        GPIO.output(INHIB_D,GPIO.LOW)
    time.sleep(0.005) #Maybe comment this out
    #Send out requests to read the voltages to each value
    values = [0]*16
    adc1.send_read_request_0()
#    time.sleep(0.0002)
    adc2.send_read_request_0()
#    time.sleep(0.0002)
    adc3.send_read_request_0()
#    time.sleep(0.0002)
    adc4.send_read_request_0()
    #sleep until adc has had time to make read
    time.sleep(1.0/DATA_RATE+0.0001)
    #retrieve values
    values[0] = adc1.retrieve_read()
#    time.sleep(0.0002)
    values[4] = adc2.retrieve_read()
#    time.sleep(0.0002)
    values[8] = adc3.retrieve_read()
#    time.sleep(0.0002)
    values[12] = adc4.retrieve_read()
    #send out next bacth of requests
#    time.sleep(0.0002)
    adc1.send_read_request_1()
#    time.sleep(0.0002)
    adc2.send_read_request_1()
#    time.sleep(0.0002)
    adc3.send_read_request_1()
#    time.sleep(0.0002)
    adc4.send_read_request_1()
    #sleep until adc has had time to make read
    time.sleep(1.0/DATA_RATE+0.0001)
    #retrieve values
    values[1] = adc1.retrieve_read()
#    time.sleep(0.0002)
    values[5] = adc2.retrieve_read()
#    time.sleep(0.0002)
    values[9] = adc3.retrieve_read()
#    time.sleep(0.0002)
    values[13] = adc4.retrieve_read()
    #send out 3rd bacth of read requests
#    time.sleep(0.0002)
    adc1.send_read_request_2()
#    time.sleep(0.0002)
    adc2.send_read_request_2()
#    time.sleep(0.0002)
    adc3.send_read_request_2()
#    time.sleep(0.0002)
    adc4.send_read_request_2()
    #sleep until adc has had time to make read
    time.sleep(1.0/DATA_RATE+0.0001)
    #retrieve values
    values[2] = adc1.retrieve_read()
#    time.sleep(0.0002)
    values[6] = adc2.retrieve_read()
#    time.sleep(0.0002)
    values[10] = adc3.retrieve_read()
#    time.sleep(0.0002)
    values[14] = adc4.retrieve_read()
    #send out last batch of read requests
#    time.sleep(0.0002)
    adc1.send_read_request_3()
#    time.sleep(0.0002)
    adc2.send_read_request_3()
#    time.sleep(0.0002)
    adc3.send_read_request_3()
#    time.sleep(0.0002)
    adc4.send_read_request_3()
    #sleep until adc has had time to make read
    time.sleep(1.0/DATA_RATE+0.0001)
    #retrieve values
    values[3] = adc1.retrieve_read()
#    time.sleep(0.0002)
    values[7] = adc2.retrieve_read()
#    time.sleep(0.0002)
    values[11] = adc3.retrieve_read()
#    time.sleep(0.0002)
    values[15] = adc4.retrieve_read()
    return values

    print("CHECK4")
#if __name__ == '__main__':
    # Boot up delay
#    t0 = time.time()
    
#    while (time.time() - t0 < 15):
#        time.sleep(1)
        
    # Initialize MQTT
    #if (MQTT_EN):
    #    broker_address = 'server.healthyaging'
    #    broker_port = 61613

        # Setup MQTT connection
    #    client = mqtt.Client(client_id='PressureMat', 
    #                         clean_session=True,
    #                         protocol=mqtt.MQTTv31) #additional parameters for clean_session, userdata, protection,
    #    client.username_pw_set('admin', 'IBMProject$')
    #    print('MQTT: Connecting to broker {0}:{1}'.format(broker_address, broker_port))
    #    client.connect(host=broker_address, port=broker_port)
    #    client.loop_start() #start the loop
    #    topic_data = 'PressureMat/raw'
    #else:
    #    client = None
    
    
    #Run the sensor
#    t_last_mqtt_msg = time.time()
#    t_mqtt_msg_period = 1
try:
    while True:
        before = time.time()
        #Read the senspor
        col0 = readColumn(C0)
        col1 = readColumn(C1)
        col2 = readColumn(C2)
        col3 = readColumn(C3)
        col4 = readColumn(C4)
        col5 = readColumn(C5)
        col6 = readColumn(C6)
        col7 = readColumn(C7)
        col8 = readColumn(C8)
        col9 = readColumn(C9)
        col10 = readColumn(C10)
        col11 = readColumn(C11)
        col12 = readColumn(C12)
        col13 = readColumn(C13)
        col14 = readColumn(C14)
        col15 = readColumn(C15)
        col16 = readColumn(C16)
        col17 = readColumn(C17)
        col18 = readColumn(C18)
        col19 = readColumn(C19)
        col20 = readColumn(C20)
        col21 = readColumn(C21)
        col22 = readColumn(C22)
        col23 = readColumn(C23)
        col24 = readColumn(C24)
        col25 = readColumn(C25)
        col26 = readColumn(C26)
        col27 = readColumn(C27)
        col28 = readColumn(C28)
        col29 = readColumn(C29)
        col30 = readColumn(C30)
        col31 = readColumn(C31)
        values = [col0,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31]
        after = time.time()
        #Print read values
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print

        dict_results = {'timestamp':time.time()}
        for i in range(16):
            result = ''
            result_disp = ''
            for j in values:
                if  j[i] < 400:
                    char = " "
                elif j[i] < 800:
                    char = "."
                elif j[i] < 1000:
                    char = "*"
                elif j[i] < 1300:
                    char = "O"
                else:
                    char = "@"
                    #char = str(j[i])
                result += char + ' '
                result_disp += char + char + '  '
            key_name = 'result{0:02d}'.format(i)
            dict_results[key_name] = j
            print('Row' + '{0:2d}'.format(i) + ' ' + result_disp)
            print('      ' + result_disp)
        print
        print
        print(after - before)
#            if (not(client is None) & (time.time() - t_last_mqtt_msg > t_mqtt_msg_period)):
#                try:
#                    msg = json.dumps(dict_results)
#                    client.publish(topic_data, msg)
#                    t_last_mqtt_msg = time.time()
#                except Exception as e:
#                    print(e)
    
    #Handle a ^C exit
except KeyboardInterrupt:
    adc1.stop_adc()
    adc2.stop_adc()
    adc3.stop_adc()
    adc4.stop_adc()
    GPIO.cleanup()
        
#    if (MQTT_EN):
#        client.loop_stop()
    
#    sys.exit(0)
