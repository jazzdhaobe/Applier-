import argparse
import json
import os
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
    if not password:
        raise SystemExit(
            "Missing password. Pass --password or set JOBAUTO_PASSWORD."
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
        "--dry-run",
        action="store_true",
        help="print the resolved settings without launching the browser",
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
    )

    if args.filters:
        nb.filter_apply(config["search"], config["experience"], config["location"], config["job_age"])
    else:
        nb.start_apply("apply")


if __name__ == "__main__":
    main()