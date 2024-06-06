from setuptools import find_packages, setup

package_name = 'live_classifier'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='accl_orin_nano2',
    maintainer_email='accl_orin_nano2@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'img_publisher      = live_classifier.webcam_pub:main',
            'img_subscriber     = live_classifier.webcam_sub:main',
            'gpu_img_publisher  = live_classifier.arguscam_pub:main',
            'gpu_img_subscriber = live_classifier.arguscam_sub:main',
            'webcam_classifier  = live_classifier.webcam_classifier:main',
        ],
    },
)
