import pymysql


def get_db_connection():
    try:
        conn = pymysql.connect(
            host="127.0.0.1",
            user="root",
            password="",
            database="hr_app_db",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
            connect_timeout=5,
        )

        print("✅ Database connected!")

        return conn

    except Exception as e:

        print(f"❌ DB Error: {e}")

        return None


def authenticate_user(
    email_or_name,
    password,
):
    try:

        conn = get_db_connection()

        if conn is None:
            return None

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    name,
                    email
                FROM users
                WHERE
                    (
                        email = %s
                        OR name = %s
                    )
                    AND password = %s
                """,
                (
                    email_or_name,
                    email_or_name,
                    password,
                ),
            )

            user = cursor.fetchone()

        conn.close()

        return user

    except Exception as e:

        print(
            f"❌ Auth Error: {e}"
        )

        return None


def get_attendance_summary():

    return {
        "total_employees": 0,
        "today_attendance": 0,
        "pending_approvals": 0,
        "absent": 0,
    }


def record_attendance(
    employee_id,
    status="present",
):

    print(
        "📝 [TEST] Attendance:",
        f"employee_id={employee_id},",
        f"status={status}",
    )

    return True


def get_today_attendance():

    return []