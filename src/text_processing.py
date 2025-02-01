import re
from datetime import datetime, timedelta
from string import punctuation
from string import digits

class New_Nic_Text( object ):
    def __init__(self) -> None:
        super().__init__()

    def datetime_format( self, in_data ):
        try:
            in_date = re.sub("[^-Z0-9]+", "",in_data)
            s_datetime = datetime.strptime(in_date, '%Y%m%d')

            return s_datetime.strftime('%Y-%m-%d')
        
        except:
            return None

    def nic_format( self , in_nic ):
        in_txt = in_nic.split(":")[-1]
        return re.sub("[^-Z0-9]+", "",in_txt)

    def name_format( self, in_name ):

        in_name = in_name.lower()
        in_name = re.sub("[^a-zA-Z0-9 ]+", "",in_name)
        if( ("name" in in_name)  ):
            return in_name.replace('name','').upper()

        else:
            return in_name.upper()

    def gender_format( self , in_name ):

        in_name = in_name.lower()

        if( "female" in in_name ):
            return "FeMale"

        elif( ("male" in in_name)  ):
            return "Male"
        else:
            return None

    def text_format( self ,  text_dict ):
        try:
            text_dict["first_name"] = self.name_format( text_dict["first_name"] )
            text_dict["gender"] = self.gender_format( text_dict["gender"] )
            text_dict["DOB"] = self.datetime_format( text_dict["DOB"] )
            text_dict["Nic_no"] = self.nic_format( text_dict["Nic_no"] )

            return text_dict

        except :
            return None
        
    def text_sim_format( self , text_dict ):
        try:
            text_dict["nic_no"] = self.nic_format( text_dict["nic_no"] )

            return text_dict

        except :
            return None


class Driving_Text( object ):
    def __init__(self) -> None:
        super().__init__()

    def license_no_format( self, in_data ):
        try:
            text = re.sub(r'[?|$|.|!]',r'',in_data)
            text = text.lstrip(digits)
            return text
        
        except:
            return None

    def nic_no_format( self , in_nic ):

        text = in_nic.lstrip(digits)
        text = re.sub('\D', '', text)
        return text

    def dob_format( self, in_dob ):

        text = in_dob.lstrip(digits)
        text = text.strip(punctuation)
        return text

    def address_format( self , in_add ):

        hypotetical = [ "8." , "8" ]

        if( in_add[:2] in hypotetical[0] ):
          in_add = in_add[2:]
        elif( in_add[:1] in hypotetical[1] ):
          in_add = in_add[1:]

        return in_add

    def text_format( self ,  text_dict ):
        try:
            text_dict["license_no"] = self.license_no_format( text_dict["license_no"] )
            text_dict["first_name"] = self.license_no_format( text_dict["first_name"] )
            text_dict["nic_no"] = self.nic_no_format( text_dict["nic_no"] )
            text_dict["DOB"] = self.dob_format( text_dict["DOB"] )
            text_dict["address_line_1"] = self.address_format( text_dict["address_line_1"] )

            return text_dict
        except:
            return None
        
    def text_sim_format( self , text_dict ):
        try:
            text_dict["id_no"] = self.license_no_format( text_dict["id_no"] )
            text_dict["nic_no"] = self.nic_no_format( text_dict["nic_no"] )

            return text_dict
        except:
            return None