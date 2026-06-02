import argparse
import json
import os
import sys
from .modules.model import ChatbotBuild
from .modules.naukri import NaukriBot

DEFAULT_SEARCH = (
    "Gen Ai, Agentic AI, AI Developer, AI Ml Engineer, AI Engineer, "
    "Software Engineer, Full stack developer, AI full stack developer, "
    "Backend Engineer, NodeJS backend Engineer"
)


def get_env(name, default=None):
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def parse_optional_int(value, default):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve_config(args):
    password = args.password or get_env("JOBAUTO_PASSWORD")
    storage_state_path = args.storage_state or get_env("JOBAUTO_STORAGE_STATE_PATH")
    storage_state_json = get_env("JOBAUTO_STORAGE_STATE")

    if getattr(args, "fresh_login", False):
        storage_state_path = None
    elif storage_state_path is None and storage_state_json:
        storage_state_path = ".auth/storage_state.json"
        try:
            storage_state_file = Path(storage_state_path)
            storage_state_file.parent.mkdir(parents=True, exist_ok=True)
            storage_state_file.write_text(storage_state_json, encoding="utf-8")
        except Exception as exc:
            raise SystemExit(
                f"Unable to write storage state from JOBAUTO_STORAGE_STATE: {exc}"
            )

    if not password and storage_state_path is None:
        raise SystemExit(
            "Missing password or storage state. Pass --password or set JOBAUTO_PASSWORD, "
            "or provide --storage-state / JOBAUTO_STORAGE_STATE_PATH / JOBAUTO_STORAGE_STATE to reuse an existing session."
        )

    save_storage_state_path = (
        args.save_storage_state
        or get_env("JOBAUTO_STORAGE_STATE_OUTPUT")
        or storage_state_path
    )

    return {
        "email": args.email,
        "username": args.email,
        "password": password,
        "number": parse_optional_int(args.number, parse_optional_int(get_env("JOBAUTO_NUMBER"), 50)),
        "search": args.search or get_env("JOBAUTO_SEARCH") or DEFAULT_SEARCH,
        "experience": parse_optional_int(
            args.experience, parse_optional_int(get_env("JOBAUTO_EXPERIENCE"), 2)
        ),
        "location": args.location or get_env("JOBAUTO_LOCATION") or "",
        "job_age": parse_optional_int(
            args.job_age, parse_optional_int(get_env("JOBAUTO_JOBAGE"), 7)
        ),
        "headless": args.headless,
        "otp": args.otp or get_env("JOBAUTO_OTP"),
        "storage_state_path": storage_state_path,
        "save_storage_state_path": save_storage_state_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Apply using NaukriBot")
    parser.add_argument("--apply", action="store_true", help="start applying to jobs")
    parser.add_argument("--train", action="store_true", help="train the chatbot model")
    parser.add_argument("--email", required=True, help="email address associated with naukri account")
    parser.add_argument("--filters", action="store_true", help="apply search filters before applying")
    parser.add_argument("--password", help="Naukri password (or use JOBAUTO_PASSWORD)")
    parser.add_argument("--search", help="override the default search keywords")
    parser.add_argument("--experience", help="minimum experience to filter by")
    parser.add_argument("--location", help="location filter")
    parser.add_argument("--job-age", help="max job age in days")
    parser.add_argument("--number", help="number of applications to attempt")
    parser.add_argument("--headless", action="store_true", help="run browser in headless mode")
    parser.add_argument("--otp", help="OTP code to use if Naukri requires a second-step verification")
    parser.add_argument(
        "--storage-state",
        help="path to a Playwright storage state JSON file to reuse as a saved session",
    )
    parser.add_argument(
        "--save-storage-state",
        help="path where the browser storage state should be written after a successful login",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved settings without launching the browser",
    )
    parser.add_argument(
        "--verify-login",
        action="store_true",
        help="only verify that login/session works, then exit",
    )
    parser.add_argument(
        "--fresh-login",
        action="store_true",
        help="ignore saved session file and login with password (local default behavior)",
    )
    args = parser.parse_args()

    if args.train:
        chb = ChatbotBuild(args.email)
        chb.train_model()
        training_data = chb.training_data
        with open(f"./jab/data/{args.email}/training_data.json", "w+") as handle:
            handle.write(json.dumps(training_data, indent=4))
        return

    config = resolve_config(args)
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    if not args.apply:
        print("Use --apply to start applying or --train to train the model")
        return

    nb = NaukriBot(
        config["email"],
        config["password"],
        config["username"],
        config["number"],
        headless=config["headless"],
        otp=config["otp"],
        storage_state_path=config["storage_state_path"],
        save_storage_state_path=config["save_storage_state_path"],
    )

    if args.verify_login:
        nb.init_browser()
        try:
            if not nb.login():
                print("Login verification failed.", file=sys.stderr)
                raise SystemExit(1)
            print("Login verification succeeded.")
            raise SystemExit(0)
        finally:
            nb.close()

    if args.filters:
        result = nb.filter_apply(
            config["search"],
            config["experience"],
            config["location"],
            config["job_age"],
        )
        if (result or {}).get("response") == "login failed":
            raise SystemExit(1)
    else:
        result = nb.start_apply("apply")
        if result is None:
            raise SystemExit(1)


if __name__ == "__main__":
    main()