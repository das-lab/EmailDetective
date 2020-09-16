import pymongo
from config import mongo_url

myclient = pymongo.MongoClient(mongo_url)
email_db = myclient['email_db']
mycol4 = email_db['email_content_3']


def get_person_by_name3(name: str, **display):
    """
    :param name:  X-From
    :param display:  
    :return:
    """
    display['_id'] = 0
    person_detail = mycol4.find({"X-From": name, "content": {"$regex": "^[^-]"}}, display)
    return person_detail


def get_person_by_name3_random(name: str, **display):
    """
    :param name:  X-From
    :param display:  
    :return:
    """
    display['_id'] = 0
    person_detail = mycol4.find({"X-From": {"$ne": name}, "content": {"$regex": "^[^-]"}}, display)
    return person_detail
