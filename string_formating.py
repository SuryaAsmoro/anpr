import re
import string

# region Global String Formating Variable

PLATE_REGEX = (r'^[A-Z]{1,2}', r'^[0-9]{1,4}', r'^[A-Z]{1,3}')
ALPHANUMERIC_REGEX = r'[^a-zA-Z0-9\s-]'

PLATE_SEGMENT_1 = re.compile(PLATE_REGEX[0])
PLATE_SEGMENT_2 = re.compile(PLATE_REGEX[1])
PLATE_SEGMENT_3 = re.compile(PLATE_REGEX[2])

ALPHANUMERIC_SEGMENT = re.compile(ALPHANUMERIC_REGEX)

#endregion

# region Mapping Dictionary for Character conversion
def correct_to_alphabet(word):
    temp = ''
    # region Switch case Module
    for c in word:
        # Zero can be Recognized as O or D
        if c == '0':
            temp += 'Q'
            continue
        elif c == '1':
            temp += 'I'
            continue
        elif c == '2':
            temp += 'Z'
            continue
        elif c == '4':
            temp += 'A'
            continue
        elif c == '6':
            temp += 'G'
            continue
        elif c == '7':
            temp += 'T'
            continue
        elif c == '8':
            temp += 'B'
            continue
        elif c == '9':
            temp += 'J'
            continue
        else:
            temp += c
    # endregion
    return temp

def correct_to_numeric(word):
    temp = ''
    # region switch cases module
    for c in word:
        if c == 'A':
            temp += '4'
            continue
        elif c == 'B':
            temp += '8'
            continue
        elif c == 'C':
            temp += '3'
            continue
        elif c == 'D':
            temp += '0'
            continue
        elif c == 'G':
            temp += '6'
            continue
        elif c == 'I':
            temp += '1'
            continue
        elif c == 'J':
            temp += '9'
            continue
        elif c == 'L':
            temp += '4'
            continue
        # Zero can be Recognized as O or D
        elif c == 'O':
            temp += '0'
            continue
        elif c == 'Q':
            temp += '0'
            continue
        elif c == 'R':
            temp += '4'
            continue
        elif c == 'S':
            temp += '8'
            continue
        elif c == 'T':
            temp += '7'
            continue
        elif c == 'U':
            temp += '0'
            continue
        elif c == 'Y':
            temp += '7'
            continue
        elif c == 'Z':
            temp += '2'
            continue
        else:
            temp += c
    # endregion
    return temp

# endregion

# region License Compile Formats
# Function to check wheter the text is in right format
def license_complies_format_v1(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """

    # region Indonesian Format
    
    # Words
    words_1, words_2, words_3 = '', '', ''
    
    # clear the non alphanumeric
    text = remove_nonalphanumeric(text)
    mutable_text = text

    # Make sure 3 words in the array are in the pattern
    if  PLATE_SEGMENT_1.search(mutable_text):

        # Get the match object from regex
        words_1 = PLATE_SEGMENT_1.search(mutable_text).group()

        print(words_1)
        
        # Remove the matched string
        mutable_text = re.sub(PLATE_SEGMENT_1, '', mutable_text)

        print(mutable_text)

    if  PLATE_SEGMENT_2.search(mutable_text):

        # Get the match object from regex
        words_2 = PLATE_SEGMENT_2.search(mutable_text).group()

        print(words_2)
        
        # Remove the matched string
        mutable_text = re.sub(PLATE_SEGMENT_2, '', mutable_text)

        print(mutable_text)

    if  PLATE_SEGMENT_3.search(mutable_text):

        # Get the match object from regex
        words_3 = PLATE_SEGMENT_3.search(mutable_text).group()

        print(words_3)
        
        # Remove the matched string
        mutable_text = re.sub(PLATE_SEGMENT_3, '', mutable_text)

        print(mutable_text)
    
    else:
        # return false if above condition not fullfiled
        return False
    
    
    return True
    
    
    
    # endregion

    # region UK Format
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False
    """
    # endregion

# Function to check wheter the text is in right format of Common Plate
def license_compile_format_ideal(text):

    # remove non alphanmeric character except Hypen and Whitespace
    text = remove_nonalphanumeric(text)

    # split string into array of words separated by space
    word_array = text.upper().split(" ")
    word_array_clear = []

    # Clean double or triple whitespace
    for word in word_array:
        if word != '' and word != ' ':
            word_array_clear.append(word)
    
    # If the the final word count == 3
    if (len(word_array_clear)) == 3:
        return True
    
    return False

# Function to check wheter the text is in right format of Military Plate
def license_compile_format_noncivil(text):

    # remove non alphanmeric character except Hypen and Whitespace
    text = remove_nonalphanumeric(text)

    # Hypen 
    if '-' in text:

        # split string into array of words separated by hypen
        word_array = text.upper().split("-")
        word_array_clear = []

        # Clean double or triple whitespace
        for word in word_array:
            if word != '' and word != ' ':
                word_array_clear.append(word)
        
        # If the the final word count == 2
        if (len(word_array_clear)) == 2:
            if len(word_array_clear[1]) == 2:
                return True
        
    return False

# Function to check wheter the text is in right format of Special Government Plate
def license_compile_format_special(text):

    # remove non alphanmeric character except Hypen and Whitespace
    text = remove_nonalphanumeric(text)

    # split string into array of words separated by space
    word_array = text.upper().split(" ")
    word_array_clear = []

    # Clean double or triple whitespace
    for word in word_array:
        if word != '' and word != ' ':
            word_array_clear.append(word)
    
    # If the the final word count == 2
    if (len(word_array_clear)) == 2:

        # Special Case of government vehicle
        if correct_to_alphabet(word_array_clear[0]) == "RI":
            return True
        
    return False

# endregion

# Function to Correct the format of the ocr result
def format_license_ideal(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    # Prepare memory for final words
    formatted_plate = ""
    extra_score = 0

    # remove non alphanmeric character except Hypen and Whitespace
    text = remove_nonalphanumeric(text)

    # split string into array of words separated by space
    word_array = text.upper().split(" ")
    word_array_clear = []

    # Clean double or triple whitespace
    for word in word_array:
        if word != '' and word != ' ':
            word_array_clear.append(word)
            

    # GUARD for ideal License
    if (len(word_array_clear)) == 3:

        # Correct incorrect detected plate sequences
        word_array_clear[0] = correct_to_alphabet(word_array_clear[0])

        # Remove I from first letter if Line is detected as Alphabet
        if word_array_clear[0][0] == 'I':
            temp = list(word_array_clear[0]).pop(0)
            word_array_clear[0] = temp

        word_array_clear[1] = correct_to_numeric(word_array_clear[1])
        word_array_clear[2] = correct_to_alphabet(word_array_clear[2])

        # Calculate extra score for words with most common sequence
        if len(word_array_clear[0]) < 3:
            extra_score += (10 - (len(word_array_clear[0]) - 1) * 10) # Single Digit got 10 Extra Score
        if len(word_array_clear[1]) == 4:
            extra_score += 10 # Full 4 Digits get 10 Extra score
        if len(word_array_clear[2]) < 4:
            extra_score += (len(word_array_clear[2]) - 1) * 10 # 2 digits got 10 extra score, and 3 Digits got 20 extra score


        formatted_plate = word_array_clear[0] + ' ' +word_array_clear[1] + ' ' + word_array_clear[2]
            
    return formatted_plate, extra_score

def format_license_noncivil(text):

    # remove non alphanmeric character except Hypen and Whitespace
    text = remove_nonalphanumeric(text)

    return text, 0

def format_license_special(text):

    # remove non alphanmeric character except Hypen and Whitespace
    text = remove_nonalphanumeric(text)

    return text, 0

# Function to Remove non-Alphanumeric from string
def remove_nonalphanumeric(text):
        
    # replace character matches in Regular expression with empty ''
    # Regular Expression : [^a-zA-Z0-9\s-] means : non-Aplhanumeric except whitespace
    clearString = re.sub(ALPHANUMERIC_SEGMENT, '', text)

    return clearString

def remove_space(text):

    clearString = text.replace(" ", "")

    return clearString


