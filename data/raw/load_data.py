
import wget
import zipfile
import glob2
import os

def get_data(src, dst):
    """
    Downloads a file from a remote source src to a local destination dst.
    """
    remote_url = src
    local_dir = dst
    wget.download(remote_url, local_dir)


def unzip_data(my_dir):
    """Unpacks the first zip file in a directory"""
    local_dir = my_dir
    file = glob2.glob(f'{local_dir}/*.zip')[0]
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(my_dir)

if __name__ == '__main__':
  print ('starting download')
  get_data('https://dsti-aws-class-ak.s3.eu-west-1.amazonaws.com/kagglecatsanddogs_3367a.zip',
          '/content/classification/data/raw')
  unzip_data('/content/classification/data/raw')

  # Make http request for remote file data

  