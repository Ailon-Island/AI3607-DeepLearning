@echo off
for /d %%d in (*) do (
	echo %%d
	ren %%d\models\*.pth *.pkl
)
pause