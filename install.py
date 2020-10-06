import sys
import subprocess
import os




def main():
    assert sys.version_info.major == 3, "Python version of 3.6 and above is required"
    assert sys.version_info.minor >=6, "Python version of 3.6 and above is required"

    cur_working_path = os.getcwd()

    #install requirements python
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", os.path.join(cur_working_path, "requirements.txt")])

    #install tinyobj
    tiny_obj_path = os.path.join(cur_working_path, "Resources", "external" ,"tinyobjloader")
    os.chdir(tiny_obj_path)
    subprocess.check_call([sys.executable, os.path.join(tiny_obj_path, "setup.py"), "install"])
    os.chdir(cur_working_path)

    #install openexr
    open_exr_path = os.path.join(cur_working_path, "Resources", "external", "openexr")
    os.chdir(open_exr_path)
    open_exr_file = "OpenEXR-1.3.2-cp3{}-cp3{}{}-win_amd64.whl".format(str(sys.version_info.minor),str(sys.version_info.minor), "m" if sys.version_info.minor!=8 else "")
    subprocess.check_call([sys.executable, "-m", "pip","install", open_exr_file])
    os.chdir(cur_working_path)
    
    #download model

    import requests

    def download_file_from_google_drive(id, destination):
        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = get_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        save_response_content(response, destination)    

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
    g_id = r"1_SN4kIfJBbhPisl0kIij_S8Cf2vW_lQK"
    filename = os.path.join(cur_working_path, "Resources", "data", "model.onnx")

    download_file_from_google_drive(g_id, filename)



if __name__ == "__main__":
    main()