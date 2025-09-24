import classNames from "classnames";

type ExampleLinkProps = {
  className?: string;
  href: string;
  title: string;
  body: string;
  icon: string;
};

export default function ExampleLink({
  className,
  href,
  title,
  body,
  icon,
}: ExampleLinkProps) {
  return (
    <a className={className} href={href} target="_blank">
      <div className="group relative flex items-start p-4 gap-4 h-auto sm:h-36 w-auto sm:w-[500px] bg-white rounded-lg border border-gray-300 shadow hover:shadow-xl hover:border-purple-400 transition duration-200 text-left">
        <div className="flex flex-none items-center justify-center w-28 h-28 rounded-lg bg-gray-100 text-3xl border border-gray-300 group-hover:border-purple-400 group-hover:bg-purple-50 group-hover:text-purple-600 transition duration-200">
          {icon}
        </div>
        <div className="flex flex-col gap-1">
          <div className="text-gray-900 font-semibold">{title}</div>
          <div className="text-gray-500 text-sm">{body}</div>
        </div>
        <div className="absolute right-0 top-0 p-2 text-purple-600 opacity-0 group-hover:opacity-100 transition duration-200">
          ↗
        </div>
      </div>
    </a>
  );
}
