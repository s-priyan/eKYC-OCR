
from collections import OrderedDict

from src.video_capture import VideoCaptureAsync
from src.video_process import VideoProcessorAsync
from src.visitortags_api_access_functions import api_connector


class Tenant:
    """
    Class encompassing objects and methods corresponding to a particular tenant
    """

    def __init__(self, tenant_id, cameras, visitor_db):
        """
        Initiates a tenant object
        :param int tenant_id: id corresponding to a tenant should be unique
        :param str name: name of the tenant
        :param Json cameras: Json array containing camera json object. each camera json object has attributes id, name
        and url
        :param int max_db_size: maximum size of db entries
        """
        self.tenant_id = tenant_id
        self.visitor_db = visitor_db

        self.video_getters = OrderedDict()
        self.video_processors = OrderedDict()

        for cam_id in cameras:
            cam_instance = VideoCaptureAsync(cam_id=cam_id, url=cameras[cam_id])

            if cam_instance.isOpened():
                self.video_getters[cam_id] = cam_instance
                self.video_processors[cam_id] = VideoProcessorAsync(self.video_getters[cam_id], self.visitor_db)
            else:
                print("camera ", cam_id, " is not accessible. Please check IP and credentials")
                api_connector.post_cam_offline(cam_id)
                del cam_instance

    def initiate(self):
        """
        Starts the async video getters and video processors
        :return: None
        """
        for cam_id in self.video_getters:
            self.video_getters[cam_id].start()
            self.video_processors[cam_id].start()

    def interrupt(self):
        """
        Stops the video processors only (to avoid corruption when adding new entry)
        :return: None
        """
        for cam_id in self.video_processors:
            self.video_processors[cam_id].stop()

    def resume(self):
        """
        Restarts the stopped video processors
        :return: None
        """
        for cam_id in self.video_processors:
            self.video_processors[cam_id].start()

    def update_database(self, uid, vector, request_type):
        """
        Internal function to update the vector database and user record
        :param uid: unique id corresponding to the user
        :param vector: face vector as text
        :param request_type: 0-add, 1-update, 2-delete
        :return: Status
        """

        self.interrupt()
        if request_type == 0:
            r = self.visitor_db.add(uid, vector)
        elif request_type ==1:
            r = self.visitor_db.update(uid, vector)
        else:
            r = self.visitor_db.delete(uid)

        self.resume()
        return r

    def update_camera(self, cam_id, url, request_type):
        """
        Internal function to update a camera
        :param cam_id: id of the camera
        :param url: url to access the camera
        :param request_type: 0-add, 1-update, 2-delete
        :return: Status (True/ False)
        """

        if request_type != 2:
            cam_instance = VideoCaptureAsync(url, cam_id)

            if not cam_instance.isOpened():
                del cam_instance
                return False

            if cam_id in self.video_getters:
                self.video_processors[cam_id].stop()
                self.video_getters[cam_id].stop()
                self.video_getters[cam_id].cap.release()
                del self.video_getters[cam_id]
                del self.video_processors[cam_id]

            self.video_getters[cam_id] = cam_instance
            self.video_processors[cam_id] = VideoProcessorAsync(self.video_getters[cam_id], self.visitor_db)
            self.video_getters[cam_id].start()
            self.video_processors[cam_id].start()
            return True
        else:
            if cam_id in self.video_getters:
                self.video_processors[cam_id].stop()
                self.video_getters[cam_id].stop()
                self.video_getters[cam_id].cap.release()
                del self.video_getters[cam_id]
                del self.video_processors[cam_id]
            else:
                return False
            return True
