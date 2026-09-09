from setuptools import find_packages, setup
from glob       import glob
from os.path    import isdir

package_name = 'threedof'

# Consider the following folders and their subfolders.
folders = ['launch', 'rviz', 'urdf', 'meshes']          # CHANGE AS NEEDED!

# Create a mapping of other files to be copied in the src->install
# build.  This is a list of tuples.  The first entry in the tuple is
# the install folder into which to place things.  The second entry is
# a list of files to place into that folder.  To avoid flattening
# folder structures, create appropriate tuples for each nested folder.
otherfiles = []
for topfolder in folders:
    for folder in [topfolder] + \
        [f for f in glob(topfolder+'/**/*', recursive=True) if isdir(f)]:
        # Grab the files in this folder and append to the mapping.
        files = [f for f in glob(folder+'/*') if not isdir(f)]
        otherfiles.append(('share/' + package_name + '/' + folder, files))


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]+otherfiles,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='Three DOF Development Code and Items',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'plothebi     = threedof.plothebi:main',
            'touchtable   = threedof.touchtable:main',
            'receivepoint = threedof.receivepoint:main',
        ],
    },
)
