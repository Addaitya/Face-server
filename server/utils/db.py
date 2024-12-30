from datetime import datetime, timedelta
from pymongo import MongoClient

class PersonCollection:
    def __init__(self, uri:str, *, db_name:str='Face', collection_name:str='Embeddings'):
        try:
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]

        except Exception as e:
            print("Error connecting to database server: \n{e}")

    def add_person(self, person:dict):
        keys = ['name', 'student_id', 'embedding']
        if not all(key in person for key in keys):
            raise ValueError("person must contain name, student_id and embedding field")
        
        person = {key:person[key] for key in keys}
        return self.collection.insert_one(person)

    def check_person(self, student_id: str)->bool:
        cnt = self.collection.count_documents({"student_id": student_id})
        if cnt:
            return True
        return False

    def search(self, embedding:list[int], index_name:str, field: str):
        '''
        Do vector search to find most relevant face

        embedding list(int): The encoded image of size 1024
        index_name str: name to the index to use of searching
        '''
        query = {
            "$vectorSearch": {
                "index": index_name,
                "limit": 1,
                "numCandidates": 5,
                "path": field,
                "queryVector": embedding,
            }
        }

        get_fields = {
            "$project": {
                '_id' : 0,
                'name' : 1,
                'student_id' : 1,
                "search_score": { "$meta": "vectorSearchScore" }
            }
        }
        try:
            result = self.collection.aggregate([
                query, 
                get_fields
            ])
            return list(result)
        except Exception as e:
            print(f'Error in searching: \n{e}')

class Attendance:
    def __init__(self, uri:str, *, db_name:str='Face', collection_name:str='Attendence'):
        try:
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]

        except Exception as e:
            print("Error connecting to database server: \n{e}")
    
    def check_one(self, student_id: str, time_stamp: datetime, **kwargs):
        '''
        is Student taken today's attendence
        '''
        
        st_datetime = datetime.combine(time_stamp.date(), datetime.min.time())
        ed_datetime = datetime.combine(time_stamp.date(), datetime.max.time())
        try:
            cnt = self.collection.count_documents({"student_id": student_id,  'time_stamp': {"$gte": st_datetime, "$lte": ed_datetime}})
            if cnt:
                return True
            return False
        
        except Exception as e:
            print(f"Error in attendence check: {e}")

    def add_many(self, rows:list):
        '''
        to add custom datetime add "datetime" key in each row
        '''
        try:
            if not rows:
                return
            verified_rows = []

            for row in rows:
                if "time_stamp" in row and "student_id" in row and not self.check_one(**row): 
                    verified_rows.append({
                        'time_stamp': row['time_stamp'], 
                        "student_id": row['student_id'],
                        "taken_on": datetime.now()
                    })
            
            if verified_rows:
                self.collection.insert_many(verified_rows)
        except Exception as e:
            print(f"Error in add_attendence:\n {e}")

    def fetch_attendance(self, student_id:str, n_days:int=7, *, curr_datetime=datetime.now()):
        '''
        Returns last n days attendence, not current day is counted in the n days. It also returns percentage of attendence taking n is total 
        '''
        try:
            ed_datetime = datetime.combine(curr_datetime.date(), datetime.max.time())
            st_datetime = datetime.combine(ed_datetime.date(), datetime.min.time()) + timedelta(days=-n_days+1)
            res = self.collection.find({"student_id": student_id, "time_stamp": {'$gte': st_datetime, '$lte': ed_datetime}}, {'_id': 0, 'time_stamp': 1, 'taken_on': 1})
            res = list(res)
            percent = (len(res) / n_days) * 100
            return res, percent

        except Exception as e:
            print(f"Error in fetch_attendence: \n{e}")

