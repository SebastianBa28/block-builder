from setuptools import find_packages, setup
from glob import glob
from os.path    import isdir


package_name = 'legobuilder'

# Create a mapping of other files to be copied in the src -> install
# build.  This is a list of tuples.  The first entry in the tuple is
# the install folder into which to place things.  The second entry is
# a list of files to place into that folder.
otherfiles = [
    ('share/' + package_name + '/launch', glob('launch/*')),
    ('share/' + package_name + '/rviz',   glob('rviz/*')),
    ('share/' + package_name + '/urdf',  glob('urdf/*')),
    ('share/' + package_name + '/meshes', glob('meshes/*')),
]

folders = ['launch', 'rviz', 'urdf', 'meshes']
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
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]+otherfiles,
    install_requires=['setuptools', 'open3d'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='Package demonstrating the camera and detector code',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector      = legobuilder.vision.detector:main',
            'brain         = legobuilder.brain.brain:main',
            'manipulator   = legobuilder.kinematics.manipulator:main',
            'ik_solver     = legobuilder.kinematics.ik_solver:main'
        ],
    },
)
