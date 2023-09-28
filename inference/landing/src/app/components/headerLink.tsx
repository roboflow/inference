import classNames from "classnames";

type HeaderLinkProps = {
  className: string;
  label: string;
  href: string;
  theme: "light" | "dark";
  icon: string;
};

const styles = {
  light: "text-gray-700 bg-white w-72",
  dark: "text-white w-72",
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
        "px-4 py-3 rounded-lg flex items-center border-gray-200 shadow"
      )}
      href={href}
    >
      <div className="flex items-center flex-none">
        <i
          className={
            "far fa-" +
            { icon } +
            "w-9 h-9 flex items-center justify-center rounded"
          }
        ></i>
        <div className="font-medium text-base">{label}</div>
        {/* <i className="far fa-arrow-right"></i> */}
      </div>
    </a>
  );
}
