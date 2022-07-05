import requests

def get(path, params=None):
    # make HTTP GET request to path
    headers = {"api-key":"3a127768fda6f3c3eb66314c40915683"}
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    
    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string
    
    return r

if __name__ == '__main__':
   baseUrl = 'http://www.tng-project.org/api/'
   r = get(baseUrl)
   print(r)
