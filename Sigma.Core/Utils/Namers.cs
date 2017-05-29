/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.Utils
{
    /// <summary>
    /// A common namer interface for static and dynamic naming of things. Any things.
    /// </summary>
    public interface INamer
    {
        /// <summary>
        /// Get the name using a certain registry (and the corresponding resolver) for a certain sender.
        /// Note: The sender may be used to get extra 
        /// </summary>
        /// <param name="registry">The parameter registry.</param>
        /// <param name="resolver">The resolver to the parameter registry.</param>
        /// <param name="sender">The sender of this naming request.</param>
        /// <returns>The name using the givne information.</returns>
        string GetName(IRegistry registry, IRegistryResolver resolver, object sender);
    }

    /// <summary>
    /// A static namer using a ... static name.
    /// </summary>
    [Serializable]
    public class StaticNamer : INamer
    {
        private readonly string _name;

        /// <summary>
        /// Create a static namer for a certain name.
        /// </summary>
        /// <param name="name">The name.</param>
        public StaticNamer(string name)
        {
            if (name == null) throw new ArgumentNullException(nameof(name));

            _name = name;
        }

        /// <inheritdoc />
        public string GetName(IRegistry registry, IRegistryResolver resolver, object sender)
        {
            return _name;
        }

        /// <summary>Returns a string that represents the current object.</summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString()
        {
            return _name;
        }
    }

    /// <summary>
    /// Create a dynamic namer using a certain lambda function for <see cref="INamer.GetName"/>.
    /// </summary>
    [Serializable]
    public class DynamicLambdaNamer : INamer
    {
        private readonly Func<IRegistry, IRegistryResolver, object, string> _nameFunction;

        /// <summary>
        /// Create a dynamic namer using the lambda function <see cref="INamer.GetName"/>
        /// </summary>
        /// <param name="nameFunction">The name function</param>
        public DynamicLambdaNamer(Func<IRegistry, IRegistryResolver, object, string> nameFunction)
        {
            if (nameFunction == null) throw new ArgumentNullException(nameof(nameFunction));

            _nameFunction = nameFunction;
        }

        /// <inheritdoc />
        public string GetName(IRegistry registry, IRegistryResolver resolver, object sender)
        {
            return _nameFunction.Invoke(registry, resolver, sender);
        }
    }

    /// <summary>
    /// An dynamic namer using individual parameters as items in a format string.
    /// </summary>
    [Serializable]
    public class DynamicItemisedNamer : INamer
    {
        private readonly string _formatString;
        private readonly string _embeddedFormatString;
        private readonly string[] _parameterIdentifiers;
        private readonly object[] _bufferParameters;

        /// <summary>
        /// Create a dynamic itemised namer using a format string and parameter identifiers (which will be resolved to the given values).
        /// Note: Parameter order is preserved.
        /// </summary>
        /// <param name="formatString">The format string.</param>
        /// <param name="parameterIdentifiers">The parameter identifiers.</param>
        public DynamicItemisedNamer(string formatString, params string[] parameterIdentifiers)
        {
            if (formatString == null) throw new ArgumentNullException(nameof(formatString));
            if (parameterIdentifiers == null) throw new ArgumentNullException(nameof(parameterIdentifiers));

            _formatString = formatString;
            _parameterIdentifiers = parameterIdentifiers; // not sure, maybe copy?
            _bufferParameters = new object[parameterIdentifiers.Length];

            string[] embeddedParameters = new string[parameterIdentifiers.Length];

            for (var i = 0; i < embeddedParameters.Length; i++)
            {
                embeddedParameters[i] = "{" + parameterIdentifiers[i] + "}";
            }

            _embeddedFormatString = string.Format(_formatString, args: embeddedParameters);
        }

        /// <inheritdoc />
        public string GetName(IRegistry registry, IRegistryResolver resolver, object sender)
        {
            for (int i = 0; i < _parameterIdentifiers.Length; i++)
            {
                _bufferParameters[i] = resolver.ResolveGetSingle<object>(_parameterIdentifiers[i]);
            }

            string name = string.Format(_formatString, _bufferParameters);

            for (var i = 0; i < _bufferParameters.Length; i++)
            {
                _bufferParameters[i] = null;
            }

            return name;
        }

        /// <summary>Returns a string that represents the current object.</summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString()
        {
            return _embeddedFormatString;
        }
    }

    /// <summary>
    /// A utility collection for various static and dynamic namers.
    /// </summary>
    public static class Namers
    {
        /// <summary>
        /// A static namer using a ... static name.
        /// </summary>   
        public static INamer Static(string name)
        {
            return new StaticNamer(name);
        }

        /// <summary>
        /// Create a dynamic namer using the lambda function <see cref="INamer.GetName"/>
        /// </summary>
        /// <param name="nameFunction">The name function</param>
        public static INamer Dynamic(Func<IRegistry, IRegistryResolver, object, string> nameFunction)
        {
            return new DynamicLambdaNamer(nameFunction);
        }

        /// <summary>
        /// Create a dynamic itemised namer using a format string and parameter identifiers (which will be resolved to the given values).
        /// Note: Parameter order is preserved.
        /// </summary>
        /// <param name="formatString">The format string.</param>
        /// <param name="parameterIdentifiers">The parameter identifiers.</param>
        // TODO fix this attr, for some reason can't be found [StringFormatMethod("formatString")]
        public static INamer Dynamic(string formatString, params string[] parameterIdentifiers)
        {
            return new DynamicItemisedNamer(formatString, parameterIdentifiers);
        }
    }
}
