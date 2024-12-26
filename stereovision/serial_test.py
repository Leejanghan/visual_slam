import serial

# 아두이노와 연결
def initialize_imu(port='/dev/ttyUSB0', baudrate=115200):
    ser = serial.Serial(port, baudrate, timeout=1)
    print(f"Connected to IMU on {port} at {baudrate} baud.")
    return ser

def read_yaw_from_imu(serial_connection):
    line = serial_connection.readline().decode('utf-8').strip()
    try:
        _, _, _, yaw = map(float, line.split())
        return yaw
    except ValueError:
        print("Invalid IMU data received.")
        return None

# IMU 초기화
imu_serial = initialize_imu(port='COM4')  # Windows에서는 COM 포트를 지정, Linux에서는 '/dev/ttyUSB0'

# IMU 데이터 읽기
while True:
    yaw = read_yaw_from_imu(imu_serial)
    if yaw is not None:
        print(f"Yaw: {yaw:.2f} degrees")
