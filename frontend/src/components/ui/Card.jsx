import React from "react";

export const Card = ({ children, className = "", ...props }) => {
  return (
    <div
      className={`bg-white shadow rounded-xl border border-gray-200 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};
