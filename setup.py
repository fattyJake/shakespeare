from setuptools import setup

setup(
	name="shakespeare",
	version="2.5.0",
	author="William Kinsman, Chenyu (Oliver) Ha, Harshal Samant, Yage (Jacob) Wang",
	author_email='''
					wkinsman@inovalon.com;
					cha@inovalon.com;
					hsamant@inovalon.com;
					ywang2@inovalon.com''',
	packages=["shakespeare"],
	install_requires=['matplotlib','sklearn','numpy','pandas','pyodbc','scipy','xgboost'],
	include_package_data=True,
	url="inovalon.com",
	description='''A python package designed to detect CARA condition gaps'''
)