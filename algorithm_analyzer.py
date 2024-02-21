import time

# Create Pin Point Key Dictionary
pin_points = {}

# Set Delta time start pin Point
def set_start_pin(key, show_mean = False):

    # Find if the key already in the Dictionary
    if key in pin_points:

        # Save the previous value before adding the new value
        prev_total_delta_time = pin_points[key][1]

        # Increase iteration index by 1
        iteration_index = pin_points[key][2] + 1

        # add key with start time to dictionary
        pin_points[key] = (time.time(), prev_total_delta_time, iteration_index, show_mean)

    else:
        # add key with start time to dictionary
        pin_points[key] = (time.time(), 0, 1, show_mean)
    
# Set Delta time end pin Point
def set_end_pin(key):
    delta_time = -1

    # Compare the input key with the key Dictionary
    if key in pin_points:

        # calculate delta time
        delta_time = time.time() - pin_points[key][0]

        # print/Write the delta time
        print(f'Execution time of {key}: {delta_time} seconds')

        # If show_mean parameter set to True
        if pin_points[key][3]:
            
            # add previous total delta time with current delta time
            total_delta_time = pin_points[key][1] + delta_time

            # mean = (previous value + current value) /2
            time_mean = (total_delta_time)/pin_points[key][2]

            # Save current delta time as previous delta time in Dictionary
            pin_points[key] = (pin_points[key][0], total_delta_time, pin_points[key][2], pin_points[key][3])

            # print/Write the delta time
            print(f'Average Execution time of {key}: {time_mean} seconds')

        # delete the used Pin Point
        # del(pin_points[key])

    else:
        # throw an error "Key not initialized"
        raise Exception("Key not initialized")

def get_average_execution_time(key):
    return pin_points[key][1]/pin_points[key][2]




