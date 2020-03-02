import json
from itertools import permutations
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf, collect_list
from pyspark.sql.functions import UserDefinedFunction, col
from operator import itemgetter

## Utility function to sort a word
def sortWord(word):
    return ''.join(sorted(word))

## populate freq_dict values from 0->9999
def updateFreqDict():
    global FREQ_DICT
    for k,v in FREQ_DICT.items():
        if v==0:
            FREQ_DICT[k] = MAX_SCORE

## Create dataframe from the input images json (jumbled_images.json)
def createInputDf(input_file):
    print("----- Creating jumbled words df... -----")
    jumbled_json = json.load(open(input_file, "r"))
    jumbled_flat = jumbled_json["inputs"]
    jumbled_fields = [
                        StructField('image_id', LongType(), False),
                        StructField('word', StringType(), False),
                        StructField('circled_spots', ArrayType(IntegerType(),False)),
                        StructField('solution_segments', ArrayType(IntegerType(), False), False)
                    ]
    jumbled_schema = StructType(jumbled_fields)
    jumbled_images_df = spark.createDataFrame(jumbled_flat, jumbled_schema)
    print("----- Jumbled words df created -----")
    return jumbled_images_df

## Find all anagrams of the given "word" in the dictionary
## return a dictionary of anagram:freq
def findAnagramsUDF(word):
    res = {}
    for key,value in FREQ_DICT.items():
        if((len(key) == len(word)) and sortWord(key)==sortWord(word)):
            res[key] = value
    return res

## get circled letters from the anagrams
## return a dictionary of anagram_word: circled_letters 
def getCircledSpotsLetters(anagram_dict, circled_spots):
    circled_dict = {}
    for key,v in anagram_dict.items():
        letter_list = []
        for i in range(len(circled_spots)):
            letter_list.append(key[circled_spots[i]])
        circled_dict[key] = ''.join(letter_list)
    return circled_dict

## aggregate circled letters for puzzles
## return a string containing all circled letters
def aggregateCircledLetters(participants):
    res = ""
    for val in participants:
       for k,v in val.items():
           res+=v
    return res

## given a list of letters, returns all permutations of length wordLen
## letters: string containig the circled letters from the anagrams
## wordLen: length of the words to be created from "letters"
def createAllPerms(letters, wordLen):
    perms = set(''.join(p) for p in permutations(letters, r=wordLen))
    return list(perms)


## create unique and sorted list of strings from the freq_dict
## perms: list of strings
## returns a list of unique permutation words found in the dictionary
def validateFromDict(perms):
    unique_perms = set()
    for string in perms:
        if string in FREQ_DICT:
            unique_perms.add(string)
    unique_perms_list = list(unique_perms)
    return unique_perms_list

## given a list of dictionaries (res) and a word, find if "word"
## is already existing in the list
def checkIfAlreadySeen(res, word):
    seen_words = [val for d in res for val in d.values()]
    if word in seen_words:
        return True
    return False

## remove all letters already used in another word, included in the solution
def removeLetters(letters, word):
    for l in word: 
        letters = letters.replace(l, "", 1)
    return letters

# Following function contains the logic for creating the final colelction of words
# Parameters: segments: segment lengths of input jumbled images, letters: circled letters, image_id
def finalSolution(segments, letters,image_id):
    print("----- Finding results for image: "+str(image_id))
    res = [] #[{segment_len_value: "highest_scored_word"}]
    all_perms_segments = []
    start = 0
    ans_dict = {}
    for val in segments:
        all_perms = createAllPerms(letters, val)
        valid_perms_list = validateFromDict(all_perms)
        word = ""
        word_score = MAX_SCORE
        for perm in valid_perms_list:
            if not checkIfAlreadySeen(res, perm):
                curr_score = FREQ_DICT[perm]
                if word_score > curr_score:
                    word_score = curr_score
                    word = perm
            else:
                continue
        val_dict = {}
        val_dict[val] = word
        res.append(val_dict)
        letters = removeLetters(letters, word)
    writeResults(res,image_id)
    return res

## write the results to a file
def writeResults(results,image_id):
    words = [val for d in results for val in d.values()]
    with open("results_greedy.txt", "a") as f:
        content = "Solution for image:"+str(image_id)+", is"": "+str(words)+"\n"
        f.write(content)
    f.close()
