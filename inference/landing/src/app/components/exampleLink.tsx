import classNames from "classnames";

type ExampleLinkProps = {
  href: string;
  title: string;
  body: string;
  icon: string;
};

export default function ExampleLink({
  href,
  title,
  body,
  icon,
}: ExampleLinkProps) {
  return (
    <a className="" href={href}>
      <div className="flex items-center p-4 gap-4 h-[143px] w-[500px] bg-white rounded-lg border border-gray-200 shadow hover:shadow-xl hover:bg-gray-100">
        <div className="flex flex-none items-center justify-center w-[111px] h-[111px] rounded-lg bg-gray-100 text-3xl">
          {icon}
        </div>
        <div className="flex flex-col gap-1">
          <div className="text-gray-900 font-semibold">{title}</div>
          <div className="text-gray-500 text-sm">{body}</div>
        </div>
        {/* <i className="far fa-arrow-right"></i> */}
      </div>
    </a>
  );
}
