from setuptools import find_packages, setup
from glob import glob

package_name = 'detectors'

# Create a mapping of other files to be copied in the src -> install
# build.  This is a list of tuples.  The first entry in the tuple is
# the install folder into which to place things.  The second entry is
# a list of files to place into that folder.
otherfiles = [
    ('share/' + package_name + '/launch', glob('launch/*')),
    ('share/' + package_name + '/rviz',   glob('rviz/*')),
    ('share/' + package_name + '/urdf',  glob('urdf/*')),
]


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]+otherfiles,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='Package demonstrating the camera and detector code',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cameracontrol = detectors.cameracontrol:main',
            'hsvtune       = detectors.hsvtune:main',
            'mapping       = detectors.mapping:main',
            'detector      = detectors.detector:main',
            'facedetector  = detectors.facedetector:main',
            'facedepth     = detectors.facedepth:main',
            'depthincenter = detectors.depthincenter:main',
            'detectaruco   = detectors.detectaruco:main',
            'writearuco    = detectors.writearuco:main',
            
            
            'manipulator   = detectors.manipulator:main'
        ],
    },
)
