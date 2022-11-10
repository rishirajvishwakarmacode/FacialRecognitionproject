# from FR2022 import faceencoding
import pymongo
import numpy as np



def transform(encoding):
    st1 = encoding.split('[')
    str2 = st1[1].split(']')
    str3 = str2[0].split()
    number_list = list(map(float, str3))
    encodingarray = np.array(number_list)
    return encodingarray


def getdata():
    try:
        name_list = []
        encodings_list = []
        retenc_list = []
        client = pymongo.MongoClient('mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false')
        db = client.sih2022
        collection = db.faceencodings
        dataset = collection.find({}, {'_id':0, 'name':1, 'encodings':1})
        for data in dataset:
            name_list.append(data['name'])
            encodings_list.append(data['encodings'])
        print('database connected')
        for encoding in encodings_list:
            faceencoding = transform(encoding)
            retenc_list.append(faceencoding)


    except:
        print('database not connected')

    return (retenc_list, name_list)

if __name__ == "__main__":
    getdata()
