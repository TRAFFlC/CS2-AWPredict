' Hidden launcher for the daily VRS scheduled task.
' Task Scheduler starts wscript.exe //B; this runs update_vrs.bat without a console window.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
repo = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = repo
sh.Run """" & repo & "\update_vrs.bat""", 0, False
