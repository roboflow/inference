import classNames from "classnames";

type HeaderLinkProps = {
  className: string;
  label: string;
  href: string;
  theme: "light" | "dark";
  icon: string;
};

const styles = {
  light:
    "text-gray-700 w-full sm:w-96 bg-white p-3 border border-gray-300 hover:border-purple-400 hover:text-purple-600",
  dark: "text-white w-64 p-4 h-14",
};

export default function HeaderLink({
  className,
  label,
  href,
  theme,
  icon,
}: HeaderLinkProps) {
  const style = styles[theme];
  return (
    <a
      className={classNames(
        className,
        styles[theme],
        "group relative flex items-center gap-3 rounded-lg shadow hover:shadow-xl transition duration-200 text-left"
      )}
      href={href}
    >
      {/* TODO: Replace placeholder emoji & → with actual FontAwesome icons from mocks */}
      {theme === "light" && (
        <div className="flex flex-none items-center justify-center w-9 h-9 border bg-gray-100 rounded group-hover:bg-purple-50 group-hover:border-purple-400 transition duration-200">
          {icon}
        </div>
      )}
      {theme === "dark" && <div className="flex-none">{icon}</div>}
      <div className="flex-grow font-medium text-base">{label}</div>
      <div className="pr-1">→</div>
    </a>
  );
}
