"""Offline regression tests for the documentation redirect checker."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from development.docs import check_redirect_destinations as redirects


class RedirectDestinationTests(unittest.TestCase):
    """Cover redirect source collection after removing the landing page."""

    def test_nav_collection_does_not_require_deleted_home_template(self) -> None:
        """Collect the retained navigation links without the deleted template."""
        destinations = redirects.collect_landing_page_destinations()

        self.assertTrue(destinations)
        self.assertTrue(
            all(source.startswith("mkdocs.yml#") for source in destinations)
        )
        self.assertIn("https://docs.roboflow.com/workflows", destinations.values())

    def test_homepage_destination_is_collected_from_front_matter(self) -> None:
        """Keep the new homepage destination in the checker input."""
        destinations = redirects.collect_front_matter_destinations()

        self.assertEqual(
            destinations["docs/index.md"],
            "https://docs.roboflow.com/deployment/self-hosted/self-hosted",
        )

    def test_checker_reaches_url_checks_without_the_home_template(self) -> None:
        """Exercise the CLI collection path without making network requests."""
        with (
            patch("sys.argv", ["check_redirect_destinations", "--json"]),
            patch.object(redirects, "fetch", return_value=(200, "")) as fetch,
            redirect_stdout(io.StringIO()),
        ):
            result = redirects.main()

        self.assertEqual(result, 0)
        self.assertTrue(fetch.called)
        checked_urls = {call.args[0] for call in fetch.call_args_list}
        self.assertIn(
            "https://docs.roboflow.com/deployment/self-hosted/self-hosted", checked_urls
        )


if __name__ == "__main__":
    unittest.main()
