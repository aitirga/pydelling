class _AbstractGidObject(object):
    """
    This class contains basic function definitions of the different GiD objects
    """
    def create_gid_bash(self) -> str:
        """
        This method should create a string that contains the bash code to generate the object in GiD
        Returns:
            Bash string
        """
        return ''
