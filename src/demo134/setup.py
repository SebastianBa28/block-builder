from setuptools import find_packages, setup
from glob import glob

package_name = 'demo134'

# Create a mapping of other files to be copied in the src -> install
# build.  This is a list of tuples.  The first entry in the tuple is
# the install folder into which to place things.  The second entry is
# a list of files to place into that folder.
otherfiles = [
    ('share/' + package_name + '/launch', glob('launch/*')),
]


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
    description='A simple 134 demo package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'goals1     = demo134.goals1:main',
            'goals2     = demo134.goals2:main',
            'plothebi   = demo134.plothebi:main',
        ],
    },
)
