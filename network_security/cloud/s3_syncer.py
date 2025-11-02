import os

class S3Sync:
    def sync_folder_to_s3(self, folder_path: str, aws_bucket_url: str):
        command = f"aws s3 sync {folder_path} {aws_bucket_url}"
        os.system(command)

    def sync_folder_from_s3(self, folder_path: str, aws_bucket_url: str):
        command = f"aws s3 sync {aws_bucket_url} {folder_path}"
        os.system(command)
