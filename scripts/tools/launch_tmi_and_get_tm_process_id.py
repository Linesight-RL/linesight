import subprocess

tm_process_id = None
tmi_process_id = int(
    subprocess.check_output(
        'powershell -executionPolicy bypass -command "& {$process = start-process $args[0] -passthru -argumentlist $args[1..($args.length-1)]; echo exit $process.id}" TMInterface.lnk'
    )
    .decode()
    .split("\r\n")[1]
)

print(f"Found {tmi_process_id=}")

tm_processes = list(
    filter(
        lambda s: s.startswith("TmForever"),
        subprocess.check_output("wmic process get Caption,ParentProcessId,ProcessId").decode().split("\r\n"),
    )
)
for process in tm_processes:
    name, parent_id, process_id = process.split()
    parent_id = int(parent_id)
    process_id = int(process_id)
    if parent_id == tmi_process_id:
        tm_process_id = process_id
assert tm_process_id is not None
print(f"Found {tm_process_id=}")
