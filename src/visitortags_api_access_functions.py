import requests
from requests.auth import HTTPBasicAuth
import time
import math

from src.user_records import UserRecords
import json
from src.config_parser import conf

server_conf = conf.server_configurations
DEBUG = conf.debug_params.api_access_print
user_record_buffer = conf.hyper_params.user_record_buffer


class VisitorTagAPIAccess:
    """
    Class to be used for connecting with Visitor tags API endpoints.
    """
    def __init__(self, server_config):
        """
        IInitiates the VisitorTagAPIAccess class
        :param server_config: Config object with urls of the endpoints and credentials
        """

        self.auth_url = server_config.authentication_url
        self.load_visitor_url = server_config.load_visitor_url
        self.load_cam_url = server_config.load_cam_url
        self.post_visitor_url = server_config.post_visitor_url
        self.cam_offline_url = server_config.cam_offline_url

        self.username = server_config.username
        self.password = server_config.password
        self.client_id = server_config.client_id

        r = self._get_access_token("password")
        if not r:
            _ = self._get_access_token("password")

    def _get_access_token(self, grant_type):
        """
        internal function used to get/ refresh the access token. Once called will update the existing access token of
        the class
        :param grant_type: password/refresh_token
        :return: True if success; False else
        """

        authentication = HTTPBasicAuth(self.client_id, "")
        if grant_type == "password":

            params = {
                "username": self.username,
                "password": self.password,
                "grant_type": grant_type
            }

        elif grant_type == "refresh_token":

            params = {
                "refresh_token": self.refresh_token,
                "grant_type": grant_type
            }
        else:
            raise ValueError("Invalid grant type")

        response = requests.post(
            self.auth_url,
            auth=authentication,
            data=params
        )
        if response.status_code == 200:
            r = response.json()
            self.access_token = r["access_token"]
            self.token_expire_time = time.time() + r["expires_in"]
            self.refresh_token = r["refresh_token"]
            return True
        return False

    def _refresh_token(self):
        """
        refreshes the current token
        :return: None
        """
        r = self._get_access_token("refresh_token")
        if not r:
            _ = self._get_access_token("password")

    def _load_one_page_of_db(self, database, page_no):
        """
        Accesses the database API (visitor/camera) and gets one fetches from the API endpoint
        :param database: visitor/camera
        :param page_no: page number to be fetched
        :return: Success Status (True/False), content of the page, total pages, total elements (in db)
        """
        if time.time() >= self.token_expire_time:
            self.refresh_token()

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.access_token
        }

        params = {"pageNo": page_no}

        if database == "visitor":
            url = self.load_visitor_url
        elif database == "cam":
            url = self.load_cam_url
        else:
            raise ValueError("Invalid database type")

        response = requests.get(
            url,
            headers=headers,
            params=params
        )

        if response.status_code == 401:
            self.refresh_token()

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.access_token
            }

            response = requests.get(
                url,
                headers=headers,
                params=params
            )

        if DEBUG:
            print(database, page_no, response)

        if response.status_code == 200:
            r = response.json()
            #print(r.keys())
            return True, r["content"], r["meta"]["totalPages"], r["meta"]["totalElements"]

        else:
            return False, [], 0, 0

    def load_visitor_db(self):
        """
        Loads the visitor database from the endpoint
        :return: Instance of UserRecords class populated with the API data
        """

        success, content, remaining_pages, total_records = self._load_one_page_of_db("visitor", 0)
        # defines db_size as 1024 or the next largest 2's power that can accommodate total records+ buffer
        db_size = max(2**math.ceil(math.log2(total_records + user_record_buffer)), 1024)
        visitor_db = UserRecords(db_size)

        if success:
            for record in content:
                _ = visitor_db.add(record["visitorId"], record["vector"])

        for p in range(1, remaining_pages):
            success, content, remaining_pages, _ = self._load_one_page_of_db("visitor", p)
            if success:
                for record in content:
                    _ = visitor_db.add(record["visitorId"], record["vector"])

        return visitor_db

    def load_cam_db(self):
        """
        Loads the camera database from the endpoint
        :return: dictionary with cam_ids and urls
        """
        cam_records = {}
        success, content, remaining_pages, _ = self._load_one_page_of_db("cam", 0)
        if success:
            for record in content:
                cam_records[record["cameraId"]] = record["rtspUrl"]

        for p in range(1, remaining_pages):
            success, content, remaining_pages, _ = self._load_one_page_of_db("cam", p)
            if success:
                for record in content:
                    cam_records[record["cameraId"]] = record["cameraUrl"]

        return cam_records

    def post_visitor_identification(self, user_id, confidence, cam_id, timestamp):
        """
        Function used to post a new known visitor hit.
        :param user_id: user_id
        :param confidence: confidence
        :param cam_id: id of the camera where recognized
        :param timestamp: timestamp of the recognition
        :return: None
        """

        if time.time() >= self.token_expire_time:
            self.refresh_token()

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.access_token
        }

        params = {
            "visitorId": user_id,
            "confidenceValue": str(confidence),
            "camId": str(cam_id),
            "timestamp": str(timestamp)
        }
        if DEBUG:
            print(user_id, confidence)
        response = requests.post(
            self.post_visitor_url,
            headers=headers,
            data=json.dumps(params)
        )
        if DEBUG:
            print(response)
        if response.status_code == 401:
            self.refresh_token()

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.access_token
            }

            _ = requests.post(
                self.post_visitor_url,
                headers=headers,
                data=json.dumps(params)
            )

    def post_cam_offline(self, cam_id):
        """
        Function to post that a camera is offline
        :param cam_id: id of the camera that is offline
        :return: none
        """
        if time.time() >= self.token_expire_time:
            self.refresh_token()

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.access_token
        }

        url = self.cam_offline_url.format(cam_id)
        if DEBUG:
            print(cam_id, url)
        response = requests.patch(
            url,
            headers=headers,
        )
        if DEBUG:
            print(response)
        if response.status_code == 401:
            self.refresh_token()

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.access_token
            }

            _ = requests.patch(
                url,
                headers=headers,
            )


api_connector = VisitorTagAPIAccess(server_conf)
