
class IdentificationRecord:
    """
    Class used to encapsulate aspects corresponding to one identification record
    """

    def __init__(self, user_id, conf, time):
        """
        Initializes a record
        :param user_id: user id corresponding to the identification
        :param conf: confidence level of identification
        :param time: time of identification
        """
        self.user_id = user_id
        self.confidence = conf
        self.identified_time = time
